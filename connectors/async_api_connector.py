"""
Async REST API connector for TinyForgeAI.

Provides asynchronous functionality to fetch data from REST APIs and convert
responses into training samples using the standard TinyForgeAI format.

Supports:
- GET, POST, PUT, PATCH requests (async)
- JSON and XML response parsing
- Pagination (offset, cursor, page-based)
- Authentication (API key, Bearer token, Basic auth)
- Rate limiting with configurable delays
- Retry with exponential backoff
- Response caching
- Request batching
- Concurrent request execution
- JSONPath-style data extraction
"""

import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Async HTTP client - use httpx if available
_HTTPX_AVAILABLE = False
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    pass

# Import sync config classes for compatibility
from connectors.api_connector import APIConnectorConfig, PaginationConfig


class AsyncAPIConnector:
    """
    Async REST API connector for streaming training samples from HTTP APIs.

    Supports various authentication methods, pagination styles, and
    response formats to extract training data from any REST API.

    Uses httpx for async HTTP requests with connection pooling and
    automatic retry support.
    """

    def __init__(self, config: APIConnectorConfig):
        """
        Initialize the async API connector.

        Args:
            config: APIConnectorConfig instance with connection settings.

        Raises:
            ImportError: If httpx is not installed.
        """
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "Async API connector requires httpx. "
                "Install with: pip install httpx"
            )

        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, response)}
        logger.debug(f"Initialized AsyncAPIConnector for {config.base_url}")

    async def __aenter__(self) -> "AsyncAPIConnector":
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                verify=self.config.verify_ssl,
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            logger.debug("Created new httpx AsyncClient")
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed httpx AsyncClient")

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
        import xml.etree.ElementTree as ET

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

    async def _make_request_with_retry(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict],
        data: Optional[Dict],
        json_body: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Make async HTTP request with retry logic.

        Args:
            endpoint: API endpoint (relative to base_url)
            method: HTTP method
            headers: Request headers
            params: Query parameters
            data: Form data
            json_body: JSON body

        Returns:
            Parsed response dict.
        """
        client = await self._get_client()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    data=data,
                    json=json_body,
                    headers=headers,
                )

                # Check if we should retry based on status code
                if response.status_code in self.config.retry_codes:
                    if attempt < self.config.max_retries:
                        await self._wait_for_retry(attempt, response)
                        continue

                # Raise for non-retryable HTTP errors
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                return self._parse_response(response.content, content_type)

            except httpx.HTTPStatusError as e:
                # HTTP errors from raise_for_status() - don't retry non-retryable codes
                logger.error(f"HTTP error on {endpoint}: {e}")
                raise RuntimeError(f"HTTP error: {e}")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Connection errors - retry these
                last_error = e
                logger.warning(f"Connection error on {endpoint} (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries:
                    await self._wait_for_retry(attempt)
                    continue
                raise RuntimeError(f"API request failed after {attempt + 1} attempts: {e}")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response: {e}")

        raise RuntimeError(f"API request failed: {last_error}")

    async def _wait_for_retry(self, attempt: int, response=None) -> None:
        """
        Wait before retrying with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)
            response: Optional response object (for Retry-After header)
        """
        # Check for Retry-After header
        retry_after = None
        if response is not None:
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

        logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt + 1})")
        await asyncio.sleep(delay)

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request to the API.

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
        # Check cache first
        cache_key = self._get_cache_key(endpoint, method, params, json_body)
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {endpoint}")
            return cached

        accept_type = "application/xml" if self.config.response_format == "xml" else "application/json"
        headers = {
            "Accept": accept_type,
            "Content-Type": "application/json",
            **self.config.headers,
            **self.config.get_auth_headers(),
        }

        response = await self._make_request_with_retry(
            endpoint, method, headers, params, data, json_body
        )

        # Cache the response
        self._set_cached_response(cache_key, response)

        return response

    async def test_connection(self, endpoint: str = "/") -> bool:
        """
        Test if the API connection is working.

        Args:
            endpoint: Endpoint to test (default: root)

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            await self._make_request(endpoint)
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False

    async def batch_request(
        self,
        endpoint: str,
        items: List[Dict],
        batch_size: int = 10,
        method: str = "POST",
        batch_param: str = "items",
        params: Optional[Dict] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
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

            response = await self._make_request(endpoint, method, json_body=json_body)
            yield response

            await self._apply_rate_limit()

    async def fetch_concurrent(
        self,
        endpoints: List[str],
        method: str = "GET",
        params: Optional[Dict] = None,
        max_concurrency: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple endpoints concurrently.

        Args:
            endpoints: List of API endpoints to fetch
            method: HTTP method
            params: Query parameters (applied to all requests)
            max_concurrency: Maximum concurrent requests

        Returns:
            List of responses in order of endpoints.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def fetch_one(endpoint: str) -> Dict[str, Any]:
            async with semaphore:
                return await self._make_request(endpoint, method, params)

        tasks = [fetch_one(ep) for ep in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {endpoints[i]}: {result}")
                processed.append({"error": str(result), "endpoint": endpoints[i]})
            else:
                processed.append(result)

        return processed

    async def fetch_data(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        pagination: Optional[PaginationConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
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
            async for item in self._fetch_single(endpoint, method, params, json_body):
                yield item
        elif pagination.style == "page":
            async for item in self._fetch_paginated_page(
                endpoint, method, params, json_body, pagination
            ):
                yield item
        elif pagination.style == "offset":
            async for item in self._fetch_paginated_offset(
                endpoint, method, params, json_body, pagination
            ):
                yield item
        elif pagination.style == "cursor":
            async for item in self._fetch_paginated_cursor(
                endpoint, method, params, json_body, pagination
            ):
                yield item

    async def _fetch_single(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Fetch data without pagination."""
        response = await self._make_request(endpoint, method, params, json_body=json_body)

        # Extract items from response (handles {"data": [...]}, {"items": [...]}, etc.)
        items = self._extract_items(response)
        for item in items:
            yield item

    async def _fetch_paginated_page(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> AsyncIterator[Dict[str, Any]]:
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

            response = await self._make_request(endpoint, method, page_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            for item in items:
                yield item

            if len(items) < pagination.page_size:
                break

            page += 1
            await self._apply_rate_limit()

    async def _fetch_paginated_offset(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> AsyncIterator[Dict[str, Any]]:
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

            response = await self._make_request(endpoint, method, offset_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            for item in items:
                yield item

            if len(items) < pagination.page_size:
                break

            offset += len(items)
            page_count += 1
            await self._apply_rate_limit()

    async def _fetch_paginated_cursor(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict],
        json_body: Optional[Dict],
        pagination: PaginationConfig,
    ) -> AsyncIterator[Dict[str, Any]]:
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

            response = await self._make_request(endpoint, method, cursor_params, json_body=json_body)
            items = self._extract_items(response)

            if not items:
                break

            for item in items:
                yield item

            # Extract next cursor
            cursor = self._extract_cursor(response, pagination.cursor_path)
            if not cursor:
                break

            page_count += 1
            await self._apply_rate_limit()

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

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting delay if configured."""
        if self.config.rate_limit_delay > 0:
            await asyncio.sleep(self.config.rate_limit_delay)

    async def stream_samples(
        self,
        endpoint: str,
        mapping: Dict[str, str],
        method: str = "GET",
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        pagination: Optional[PaginationConfig] = None,
        data_path: Optional[str] = None,
    ) -> AsyncIterator[Dict]:
        """
        Stream training samples from the API asynchronously.

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
        async for item in self.fetch_data(endpoint, method, params, json_body, pagination):
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


async def main() -> int:
    """Async CLI entry point for the REST API connector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream training samples from a REST API (async)."
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

    count = 0
    try:
        async with AsyncAPIConnector(config) as connector:
            async for sample in connector.stream_samples(
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
    sys.exit(asyncio.run(main()))
