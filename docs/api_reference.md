# API Reference — TinyForgeAI Inference Microservice

This document describes the REST API endpoints provided by TinyForgeAI inference services.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, TinyForgeAI services do not require authentication. For production deployments, consider adding authentication via a reverse proxy (nginx, Traefik) or API gateway.

---

## Endpoints

### POST /predict

Perform inference on input text.

#### Request

```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "input": "string"
}
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | Yes | The input text for inference |
| `metadata` | object | No | Optional metadata to include with request |

#### Response

```json
{
  "output": "string",
  "confidence": 0.75
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | string | The model's output/prediction |
| `confidence` | float | Confidence score (0.0 - 1.0) |

#### Example

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'
```

**Response:**

```json
{
  "output": "dlrow olleh",
  "confidence": 0.75
}
```

#### Error Responses

| Status | Description |
|--------|-------------|
| 400 | Bad Request - Invalid JSON or missing required fields |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |

---

### GET /health

Health check endpoint for liveness probes.

#### Request

```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### Response

```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Health status ("healthy") |
| `uptime_seconds` | float | Server uptime in seconds |

#### Example

```bash
curl http://localhost:8000/health
```

---

### GET /readyz

Readiness probe endpoint. Returns 200 when service is ready to accept requests.

#### Request

```http
GET /readyz HTTP/1.1
Host: localhost:8000
```

#### Response

```json
{
  "ready": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `ready` | boolean | Whether service is ready |

#### Example

```bash
curl http://localhost:8000/readyz
```

---

### GET /metrics

Basic metrics endpoint.

#### Request

```http
GET /metrics HTTP/1.1
Host: localhost:8000
```

#### Response

```json
{
  "request_count": 1247,
  "uptime_seconds": 3600.5
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `request_count` | integer | Total number of prediction requests |
| `uptime_seconds` | float | Server uptime in seconds |

#### Example

```bash
curl http://localhost:8000/metrics
```

---

### GET /openapi.json

OpenAPI 3.0 specification for the API.

#### Request

```http
GET /openapi.json HTTP/1.1
Host: localhost:8000
```

#### Response

Returns the full OpenAPI specification in JSON format.

---

### GET /docs

Interactive API documentation (Swagger UI).

#### Request

Open in browser:
```
http://localhost:8000/docs
```

---

### GET /redoc

Alternative API documentation (ReDoc).

#### Request

Open in browser:
```
http://localhost:8000/redoc
```

---

## OpenAPI Specification

```yaml
openapi: "3.0.0"
info:
  title: "TinyForgeAI Inference API"
  version: "0.1.0"
  description: "REST API for TinyForgeAI inference services"
  license:
    name: "Apache 2.0"
    url: "https://www.apache.org/licenses/LICENSE-2.0"

servers:
  - url: "http://localhost:8000"
    description: "Local development server"

paths:
  /predict:
    post:
      summary: "Perform inference"
      description: "Send input text and receive model prediction"
      operationId: "predict"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - input
              properties:
                input:
                  type: string
                  description: "Input text for inference"
                  example: "hello world"
                metadata:
                  type: object
                  description: "Optional metadata"
      responses:
        "200":
          description: "Successful prediction"
          content:
            application/json:
              schema:
                type: object
                properties:
                  output:
                    type: string
                    description: "Model output"
                  confidence:
                    type: number
                    format: float
                    description: "Confidence score (0-1)"
        "400":
          description: "Bad request"
        "500":
          description: "Server error"

  /health:
    get:
      summary: "Health check"
      description: "Returns service health status"
      operationId: "health"
      responses:
        "200":
          description: "Service is healthy"
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "healthy"
                  uptime_seconds:
                    type: number
                    format: float

  /readyz:
    get:
      summary: "Readiness check"
      description: "Returns whether service is ready"
      operationId: "readyz"
      responses:
        "200":
          description: "Service is ready"
          content:
            application/json:
              schema:
                type: object
                properties:
                  ready:
                    type: boolean

  /metrics:
    get:
      summary: "Get metrics"
      description: "Returns basic service metrics"
      operationId: "metrics"
      responses:
        "200":
          description: "Metrics response"
          content:
            application/json:
              schema:
                type: object
                properties:
                  request_count:
                    type: integer
                  uptime_seconds:
                    type: number
                    format: float
```

---

## SDK Examples

### Python

```python
import requests

# Basic prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"input": "hello world"}
)
result = response.json()
print(f"Output: {result['output']}")
print(f"Confidence: {result['confidence']:.2%}")

# With error handling
try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"input": "test"},
        timeout=5
    )
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

### JavaScript/TypeScript

```javascript
// Using fetch
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input: 'hello world' })
});

const result = await response.json();
console.log(`Output: ${result.output}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

### cURL

```bash
# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'

# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Errors

| Status Code | Meaning | Resolution |
|-------------|---------|------------|
| 400 | Bad Request | Check request body format |
| 404 | Not Found | Verify endpoint URL |
| 422 | Validation Error | Check required fields |
| 500 | Server Error | Check server logs |
| 503 | Service Unavailable | Wait and retry |

---

## Rate Limiting

Default TinyForgeAI services do not implement rate limiting. For production deployments, consider:

1. **Reverse Proxy**: Configure rate limiting in nginx/Traefik
2. **API Gateway**: Use AWS API Gateway, Kong, or similar
3. **Application Level**: Add FastAPI middleware

---

## See Also

- [Deployment Guide](../deploy/README.md)
- [Architecture](architecture.md)
- [Playground](../playground/README.md)
