"""
REST API connector for TinyForgeAI.

Provides functionality to fetch data from REST APIs and convert responses
into training samples using the standard TinyForgeAI format.

Supports:
- GET, POST, PUT, PATCH requests
- JSON and XML response parsing
- Pagination (offset, cursor, page-based)
- Authentication (API key, Bearer token, Basic auth)
- Rate limiting with configurable delays
- Retry with exponential backoff
- Response caching
- Request batching
- JSONPath-style data extraction
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from urllib.parse import urljoin, urlencode

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# HTTP client - use requests if available, fall back to urllib
_REQUESTS_AVAILABLE = False
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.error


class APIConnectorConfig:
    """Configuration for REST API connector."""

    def __init__(
        self,
        base_url: str,
        auth_type: Optional[str] = None,
        auth_value: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        rate_limit_delay: float = 0.0,
        timeout: int = 30,
        verify_ssl: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        retry_codes: Optional[List[int]] = None,
        cache_enabled: bool = False,
        cache_ttl: int = 300,
        response_format: str = "json",
    ):
        """
        Initialize API connector configuration.

        Args:
            base_url: Base URL for the API (e.g., "https://api.example.com/v1")
            auth_type: Authentication type: "api_key", "bearer", "basic", or None
            auth_value: Authentication value (API key, token, or "user:pass" for basic)
            headers: Additional headers to include in requests
            rate_limit_delay: Delay in seconds between requests (for rate limiting)
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum number of retry attempts for failed requests
            retry_backoff: Base delay in seconds for exponential backoff
            retry_codes: HTTP status codes to retry (default: [429, 500, 502, 503, 504])
            cache_enabled: Whether to cache responses
            cache_ttl: Cache time-to-live in seconds
            response_format: Expected response format: "json" or "xml"
        """
        self.base_url = base_url.rstrip("/")
        self.auth_type = auth_type
        self.auth_value = auth_value
        self.headers = headers or {}
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_codes = retry_codes or [429, 500, 502, 503, 504]
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.response_format = response_format

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth_type."""
        if not self.auth_type or not self.auth_value:
            return {}

        if self.auth_type == "bearer":
            return {"Authorization": f"Bearer {self.auth_value}"}
        elif self.auth_type == "api_key":
            return {"X-API-Key": self.auth_value}
        elif self.auth_type == "basic":
            import base64
            encoded = base64.b64encode(self.auth_value.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {}


class PaginationConfig:
    """Configuration for API pagination."""

    def __init__(
        self,
        style: str = "none",
        page_param: str = "page",
        limit_param: str = "limit",
        offset_param: str = "offset",
        cursor_param: str = "cursor",
        cursor_path: Optional[str] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
    ):
        """
        Initialize pagination configuration.

        Args:
            style: Pagination style: "none", "page", "offset", "cursor"
            page_param: Query parameter name for page number
            limit_param: Query parameter name for page size/limit
            offset_param: Query parameter name for offset
            cursor_param: Query parameter name for cursor
            cursor_path: JSONPath to extract next cursor from response
            page_size: Number of items per page
            max_pages: Maximum number of pages to fetch (None for unlimited)
        """
        self.style = style
        self.page_param = page_param
        self.limit_param = limit_param
        self.offset_param = offset_param
        self.cursor_param = cursor_param
        self.cursor_path = cursor_path
        self.page_size = page_size
        self.max_pages = max_pages


class APIConnector:
    """
    REST API connector for streaming training samples from HTTP APIs.

    Supports various authentication methods, pagination styles, and
    response formats to extract training data from any REST API.
    """

    def __init__(self, config: APIConnectorConfig):
        """
        Initialize the API connector.

        Args:
            config: APIConnectorConfig instance with connection settings.
        """
        self.config = config
        self._session = None
        self._cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, response)}

    def _get_session(self):
        """Get or create HTTP session."""
        if _REQUESTS_AVAILABLE:
            if self._session is None:
                self._session = requests.Session()
                self._session.verify = self.config.verify_ssl
            return self._session
        return None

    def _get_cache_key(
        self,
        url: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
    ) -> str:
        """Generate a cache key for the request."""
        key_data = f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}:{json.dumps(json_body or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if valid."""
        if not self.config.cache_enabled:
            return None
        if cache_key in self._cache:
            timestamp, response = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return response
            else:
                del self._cache[cache_key]
        return None

    def _set_cached_response(self, cache_key: str, response: Dict) -> None:
        """Cache a response."""
        if self.config.cache_enabled:
            self._cache[cache_key] = (time.time(), response)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    def _parse_response(self, content: bytes, content_type: str = "") -> Dict[str, Any]:
        """
        Parse response content based on format.

        Args:
            content: Raw response bytes
            content_type: Content-Type header value

        Returns:
            Parsed response as dict.
        """
        response_format = self.config.response_format

        # Auto-detect from content-type if available
        if "xml" in content_type.lower():
            response_format = "xml"
        elif "json" in content_type.lower():
            response_format = "json"

        if response_format == "xml":
            return self._parse_xml(content)
        else:
            return json.loads(content.decode("utf-8"))

    def _parse_xml(self, content: bytes) -> Dict[str, Any]:
        """
        Parse XML content to dict.

        Args:
            content: XML bytes

        Returns:
            Dict representation of XML.
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise RuntimeError("XML parsing requires xml.etree.ElementTree")

        def element_to_dict(element) -> Union[Dict, str, List]:
            """Recursively convert XML element to dict."""
            result: Dict[str, Any] = {}

            # Add attributes
            if element.attrib:
                result["@attributes"] = dict(element.attrib)

            # Process children
            children = list(element)
            if children:
                child_dict: Dict[str, List] = {}
                for child in children:
                    child_data = element_to_dict(child)
                    if child.tag in child_dict:
                        child_dict[child.tag].append(child_data)
                    else:
                        child_dict[child.tag] = [child_data]

                # Unwrap single-item lists
                for key, value in child_dict.items():
                    if len(value) == 1:
                        result[key] = value[0]
                    else:
                        result[key] = value
            elif element.text and element.text.strip():
                # Leaf node with text
                if result:  # Has attributes
                    result["#text"] = element.text.strip()
                else:
                    return element.text.strip()

            return result if result else ""

        root = ET.fromstring(content)
        return {root.tag: element_to_dict(root)}

    def _make_request_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict],
        data: Optional[Dict],
        json_body: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            url: Full URL
            method: HTTP method
            headers: Request headers
            params: Query parameters
            data: Form data
            json_body: JSON body

        Returns:
            Parsed response dict.
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if _REQUESTS_AVAILABLE:
                    session = self._get_session()
                    response = session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=data,
                        json=json_body,
                        headers=headers,
                        timeout=self.config.timeout,
                    )

                    # Check if we should retry based on status code
                    if response.status_code in self.config.retry_codes:
                        if attempt < self.config.max_retries:
                            self._wait_for_retry(attempt, response)
                            continue

                    # Raise for non-retryable HTTP errors (4xx except 429)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    return self._parse_response(response.content, content_type)
                else:
                    # Fallback to urllib
                    request_url = url
                    if params:
                        request_url = f"{url}?{urlencode(params)}"

                    body = None
                    if json_body:
                        body = json.dumps(json_body).encode("utf-8")
                    elif data:
                        body = urlencode(data).encode("utf-8")

                    req = urllib.request.Request(
                        request_url, data=body, headers=headers, method=method
                    )

                    try:
                        with urllib.request.urlopen(
                            req, timeout=self.config.timeout
                        ) as response:
                            content_type = response.headers.get("Content-Type", "")
                            return self._parse_response(response.read(), content_type)
                    except urllib.error.HTTPError as e:
                        if e.code in self.config.retry_codes:
                            if attempt < self.config.max_retries:
                                self._wait_for_retry(attempt)
                                continue
                        raise RuntimeError(f"HTTP error {e.code}: {e.reason}")

            except requests.HTTPError if _REQUESTS_AVAILABLE else urllib.error.HTTPError as e:
                # HTTP errors from raise_for_status() - don't retry non-retryable codes
                raise RuntimeError(f"HTTP error: {e}")
            except (requests.RequestException if _REQUESTS_AVAILABLE else urllib.error.URLError) as e:
                # Connection errors - retry these
                last_error = e
                if attempt < self.config.max_retries:
                    self._wait_for_retry(attempt)
                    continue
                raise RuntimeError(f"API request failed after {attempt + 1} attempts: {e}")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response: {e}")

        raise RuntimeError(f"API request failed: {last_error}")

    def _wait_for_retry(self, attempt: int, response=None) -> None:
        """
        Wait before retrying with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)
            response: Optional response object (for Retry-After header)
        """
        # Check for Retry-After header
        retry_after = None
        if response is not None and hasattr(response, "headers"):
            retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.config.retry_backoff * (2 ** attempt)
        else:
            # Exponential backoff with jitter
            delay = self.config.retry_backoff * (2 ** attempt)
            delay = delay * (0.5 + random.random())  # Add jitter

        time.sleep(delay)

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            endpoint: API endpoint (relative to base_url)
            method: HTTP method (GET, POST, PUT, PATCH)
            params: Query parameters
            data: Form data for POST/PUT
            json_body: JSON body for POST/PUT

        Returns:
            Parsed response as dict.

        Raises:
            RuntimeError: If request fails or response is invalid.
        """
        url = urljoin(self.config.base_url + "/", endpoint.lstrip("/"))

        # Check cache first
        cache_key = self._get_cache_key(url, method, params, json_body)
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        accept_type = "application/xml" if self.config.response_format == "xml" else "application/json"
        headers = {
            "Accept": accept_type,
            "Content-Type": "application/json",
            **self.config.headers,
            **self.config.get_auth_headers(),
        }

        response = self._make_request_with_retry(
            url, method, headers, params, data, json_body
        )

        # Cache the response
        self._set_cached_response(cache_key, response)

        return response

    def batch_request(
        self,
        endpoint: str,
        items: List[Dict],
        batch_size: int = 10,
        method: str = "POST",
        batch_param: str = "items",
        params: Optional[Dict] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Send batched requests to the API.

        Groups items into batches and sends them as single requests.
        Useful for APIs that support bulk operations.

        Args:
            endpoint: API endpoint
            items: List of items to send
            batch_size: Number of items per batch
            method: HTTP method (usually POST)
            batch_param: Parameter name for the batch items in request body
            params: Additional query parameters

        Yields:
            Response for each batch.
        """
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            json_body = {batch_param: batch}
            if params:
                json_body.update(params)

            response = self._make_request(endpoint, method, json_body=json_body)
            yield response

            self._apply_rate_limit()

    def test_connection(self, endpoint: str = "/") -> bool:
        """
        Test if the API connection is working.

        Args:
            endpoint: Endpoint to test (default: root)

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            self._make_request(endpoint)
            return True
        except Exception:
            return False

    def fetch_data(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        pagination: Optional[PaginationConfig] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from API with optional pagination.

        Args:
            endpoint: API endpoint to fetch from
            method: HTTP method
            params: Query parameters
            json_body: JSON body for POST requests
            pagination: Pagination configuration

        Yields:
            Individual items from API response.
        """
        if pagination is None or pagination.style == "none":
            yield from self._fetch_single(endpoint, method, params, json_body)
        elif pagination.style == "page":
            yield from self._fetch_paginated_page(
                endpoint, method, params, json_body, pagination
            )
        elif pagination.style == "offset":
            yield from self._fetch_paginated_offset(
                endpoint, method, params, json_body, pagination
            )
        elif pagination.style == "cursor":
            yield from self._fetch_paginated_cursor(
                endpoint, method, params, json_body, pagination
            )

    def _fetch_single(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data without pagination."""
        response = self._make_request(endpoint, method, params, json_body=json_body)

        # Extract items from response (handles {"data": [...]}, {"items": [...]}, etc.)
        items = self._extract_items(response)
        yield from items

    def _fetch_paginated_page(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data with page-based pagination."""
        params = params or {}
        page = 1

        while True:
            if pagination.max_pages and page > pagination.max_pages:
                break

            page_params = {
                **params,
                pagination.page_param: page,
                pagination.limit_param: pagination.page_size,
            }

            response = self._make_request(endpoint, method, page_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            yield from items

            if len(items) < pagination.page_size:
                break

            page += 1
            self._apply_rate_limit()

    def _fetch_paginated_offset(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data with offset-based pagination."""
        params = params or {}
        offset = 0
        page_count = 0

        while True:
            if pagination.max_pages and page_count >= pagination.max_pages:
                break

            offset_params = {
                **params,
                pagination.offset_param: offset,
                pagination.limit_param: pagination.page_size,
            }

            response = self._make_request(endpoint, method, offset_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            yield from items

            if len(items) < pagination.page_size:
                break

            offset += len(items)
            page_count += 1
            self._apply_rate_limit()

    def _fetch_paginated_cursor(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data with cursor-based pagination."""
        params = params or {}
        cursor = None
        page_count = 0

        while True:
            if pagination.max_pages and page_count >= pagination.max_pages:
                break

            cursor_params = {**params}
            if cursor:
                cursor_params[pagination.cursor_param] = cursor

            response = self._make_request(endpoint, method, cursor_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            yield from items

            # Extract next cursor
            cursor = self._extract_cursor(response, pagination.cursor_path)
            if not cursor:
                break

            page_count += 1
            self._apply_rate_limit()

    def _extract_items(self, response: Union[Dict, List]) -> List[Dict]:
        """Extract items list from API response."""
        if isinstance(response, list):
            return response

        # Common response structures
        for key in ["data", "items", "results", "records", "rows", "content"]:
            if key in response and isinstance(response[key], list):
                return response[key]

        # If response is a dict without items list, return as single item
        return [response]

    def _extract_cursor(
        self, response: Dict, cursor_path: Optional[str]
    ) -> Optional[str]:
        """Extract next cursor from response."""
        if not cursor_path:
            # Try common cursor locations
            for key in ["next_cursor", "cursor", "next", "nextToken", "next_page_token"]:
                if key in response and response[key]:
                    return str(response[key])
            return None

        # Navigate JSONPath-like path
        parts = cursor_path.split(".")
        current = response
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return str(current) if current else None

    def _apply_rate_limit(self):
        """Apply rate limiting delay if configured."""
        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)

    def stream_samples(
        self,
        endpoint: str,
        mapping: Dict[str, str],
        method: str = "GET",
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        pagination: Optional[PaginationConfig] = None,
        data_path: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Stream training samples from the API.

        Fetches data from the API and converts items to training samples
        using the specified field mapping.

        Args:
            endpoint: API endpoint to fetch from
            mapping: Field mapping dict with "input" and "output" keys.
                     Example: {"input": "question", "output": "answer"}
            method: HTTP method (GET, POST)
            params: Query parameters
            json_body: JSON body for POST requests
            pagination: Pagination configuration
            data_path: JSONPath to data array in response (e.g., "data.items")

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        for item in self.fetch_data(endpoint, method, params, json_body, pagination):
            # Navigate to nested data if path specified
            if data_path:
                for part in data_path.split("."):
                    if isinstance(item, dict) and part in item:
                        item = item[part]
                    else:
                        continue

            # Handle list items from nested path
            if isinstance(item, list):
                for sub_item in item:
                    yield self._item_to_sample(sub_item, mapping)
            else:
                yield self._item_to_sample(item, mapping)

    def _item_to_sample(self, item: Dict, mapping: Dict[str, str]) -> Dict:
        """Convert an API response item to a training sample."""
        if "input" not in mapping:
            raise KeyError("Mapping must contain 'input' key")
        if "output" not in mapping:
            raise KeyError("Mapping must contain 'output' key")

        input_field = mapping["input"]
        output_field = mapping["output"]

        # Support nested field access with dot notation
        input_value = self._get_nested_value(item, input_field)
        output_value = self._get_nested_value(item, output_field)

        if input_value is None:
            raise KeyError(f"Item missing required input field: '{input_field}'")
        if output_value is None:
            raise KeyError(f"Item missing required output field: '{output_field}'")

        return {
            "input": str(input_value),
            "output": str(output_value),
            "metadata": {
                "source": "api",
                "endpoint": self.config.base_url,
                "raw_item": item,
            },
        }

    def _get_nested_value(self, item: Dict, path: str) -> Any:
        """Get a value from nested dict using dot notation path."""
        parts = path.split(".")
        current = item
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


def main() -> int:
    """CLI entry point for the REST API connector."""
    parser = argparse.ArgumentParser(
        description="Stream training samples from a REST API."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="API endpoint to fetch from.",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help='JSON mapping string, e.g., \'{"input":"question","output":"answer"}\'',
    )
    parser.add_argument(
        "--method",
        default="GET",
        choices=["GET", "POST", "PUT", "PATCH"],
        help="HTTP method (default: GET).",
    )
    parser.add_argument(
        "--auth-type",
        choices=["api_key", "bearer", "basic"],
        help="Authentication type.",
    )
    parser.add_argument(
        "--auth-value",
        help="Authentication value (API key, token, or user:pass).",
    )
    parser.add_argument(
        "--pagination",
        choices=["none", "page", "offset", "cursor"],
        default="none",
        help="Pagination style.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Page size for pagination.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to fetch.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        help="Delay in seconds between requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to output.",
    )

    args = parser.parse_args()

    try:
        mapping = json.loads(args.mapping)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON mapping: {e}", file=sys.stderr)
        return 1

    config = APIConnectorConfig(
        base_url=args.base_url,
        auth_type=args.auth_type,
        auth_value=args.auth_value,
        rate_limit_delay=args.rate_limit,
    )

    pagination = None
    if args.pagination != "none":
        pagination = PaginationConfig(
            style=args.pagination,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )

    connector = APIConnector(config)

    count = 0
    try:
        for sample in connector.stream_samples(
            args.endpoint,
            mapping,
            method=args.method,
            pagination=pagination,
        ):
            print(json.dumps(sample))
            count += 1
            if args.limit and count >= args.limit:
                break
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
