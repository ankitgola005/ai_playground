---
# AI Playground Serving Engine

This module exposes a **MiniGPT-style text generation model** via a REST API using FastAPI and Uvicorn, optimized with **KV caching** for efficient generation over long sequences.
---

## Features

- REST API for text generation.
- Supports **KV cache** and **paged KV cache** for long context sequences.
- Deterministic generation when using fixed seeds.
- GPU/CPU support.

---

## Running the Server

Start the API:

```bash
python -m uvicorn ai_playground.serving.serving_engine:app --host localhost --port 8000
```

API documentation (Swagger UI) is available at:
`http://localhost:8000/docs`

---

## API Usage

**POST /generate**

**Request JSON:**

```json
{
  "prompts": ["Hello world!", "To be or not to be..."],
  "max_tokens": 50,
  "use_cache": true,
  "temperature": 1.0,
  "topk": 50
}
```

**Response JSON:**

```json
{
  "outputs": ["Hello world! ...", "To be or not to be... ..."],
  "stats": {
    "ctx_len": 20,
    "prefill_time": 0.0123,
    "decode_time": 0.0345
  }
}
```

- `outputs`: Generated text for each prompt.
- `stats`: Timing info for **prefill** (processing input prompts) and **decode** (generation) steps.
- `use_cache`: Set `true` for KV caching.

---

## Python Client Example

```python
import requests

url = "http://localhost:8000/generate"
data = {
    "prompts": ["Hello world!", "This is a test."],
    "max_tokens": 20,
    "use_cache": True
}

response = requests.post(url, json=data)
result = response.json()

for prompt, output in zip(data["prompts"], result["outputs"]):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
```

---
