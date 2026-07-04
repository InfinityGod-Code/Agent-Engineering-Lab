# Advanced Optimization for Production Prompt Systems

**Production Note #3: Performance, Cost, and Architecture Optimization**

This note covers advanced techniques for optimizing LLM-powered systems in production — reducing latency, cutting costs, improving reliability, and architecting for scale.

---

## 1. Caching Strategies

Caching is the single most impactful optimization for production LLM systems. A well-designed cache can reduce costs by 40-80% and latency by 90%+.

### Exact-Match Cache

Simple but effective for deterministic prompts:

```python
from functools import lru_cache
import hashlib
import redis

class ExactMatchCache:
    def __init__(self, redis_client=None, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self._local_cache = {}

    def _make_key(self, model: str, messages: list[dict], temperature: float) -> str:
        content = f"{model}:{json.dumps(messages, sort_keys=True)}:{temperature}"
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str:
        key = self._make_key(model, messages, temperature)
        if key in self._local_cache:
            return self._local_cache[key]
        if self.redis:
            result = self.redis.get(key)
            if result:
                self._local_cache[key] = result.decode()
                return self._local_cache[key]
        return None

    def set(self, model: str, messages: list[dict], response: str, temperature: float = 0):
        key = self._make_key(model, messages, temperature)
        self._local_cache[key] = response
        if self.redis:
            self.redis.setex(key, self.ttl, response)

    def invalidate(self, model: str = None, prompt_pattern: str = None):
        if self.redis:
            pattern = f"llm_cache:*"
            if model:
                pattern = f"llm_cache:{model}:*"
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                for key in keys:
                    self.redis.delete(key)
                if cursor == 0:
                    break
        self._local_cache.clear()

cache = ExactMatchCache(redis_client=redis.Redis())

def cached_llm_call(model: str, messages: list[dict], **kwargs) -> str:
    temperature = kwargs.get("temperature", 0)
    cached = cache.get(model, messages, temperature)
    if cached:
        return cached
    response = openai.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    content = response.choices[0].message.content
    cache.set(model, messages, content, temperature)
    return content
```

### Semantic Cache

Cache similar (not just identical) prompts:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class CachedEntry:
    embedding: np.ndarray
    response: str
    timestamp: float
    hit_count: int = 0

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 10000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.entries: list[CachedEntry] = []
        self.embedding_model = "text-embedding-3-small"

    def _get_embedding(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(model=self.embedding_model, input=text)
        return np.array(response.data[0].embedding)

    def find_similar(self, query: str) -> tuple[str, float]:
        query_emb = self._get_embedding(query)
        best_sim = 0
        best_response = None
        for entry in self.entries:
            sim = np.dot(query_emb, entry.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(entry.embedding)
            )
            if sim > best_sim:
                best_sim = sim
                best_response = entry.response
        if best_sim >= self.threshold and best_response:
            return best_response, float(best_sim)
        return None, 0

    def add(self, query: str, response: str):
        embedding = self._get_embedding(query)
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e.timestamp)
            self.entries.pop(0)
        self.entries.append(CachedEntry(
            embedding=embedding, response=response, timestamp=time.time()
        ))

    def get_or_compute(self, query: str, compute_fn: callable) -> str:
        cached, similarity = self.find_similar(query)
        if cached:
            return cached
        response = compute_fn(query)
        self.add(query, response)
        return response

semantic_cache = SemanticCache(similarity_threshold=0.92)

def get_answer(question: str) -> str:
    def compute(q: str) -> str:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            temperature=0
        )
        return resp.choices[0].message.content
    return semantic_cache.get_or_compute(question, compute)
```

### Two-Tier Cache Strategy

```python
class TieredCache:
    def __init__(self, exact_ttl: int = 3600, semantic_threshold: float = 0.95):
        self.exact = ExactMatchCache(ttl=exact_ttl)
        self.semantic = SemanticCache(similarity_threshold=semantic_threshold)
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    def get(self, query: str, model: str = "gpt-4o-mini", system_prompt: str = "") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        exact = self.exact.get(model, messages)
        if exact:
            self.stats["exact_hits"] += 1
            return exact
        semantic, sim = self.semantic.find_similar(query)
        if semantic:
            self.stats["semantic_hits"] += 1
            return semantic
        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: str, model: str = "gpt-4o-mini", system_prompt: str = ""):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        self.exact.set(model, messages, response)
        self.semantic.add(query, response)

    def get_hit_rate(self) -> dict:
        total = self.stats["exact_hits"] + self.stats["semantic_hits"] + self.stats["misses"]
        if total == 0:
            return {"total": 0, "exact": 0, "semantic": 0, "overall": 0}
        return {
            "total": total,
            "exact_hits": self.stats["exact_hits"],
            "exact_rate": self.stats["exact_hits"] / total,
            "semantic_hits": self.stats["semantic_hits"],
            "semantic_rate": self.stats["semantic_hits"] / total,
            "overall_hit_rate": (self.stats["exact_hits"] + self.stats["semantic_hits"]) / total
        }
```

---

## 2. Prompt Compression

Reduce token usage (and thus cost and latency) by compressing prompts intelligently.

### Truncation with Priority Preservation

```python
def truncate_to_fit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # Keep first and last portions (preserve intro + conclusion)
    head_ratio = 0.6
    head_count = int(max_tokens * head_ratio)
    tail_count = max_tokens - head_count
    truncated = tokens[:head_count] + tokens[-tail_count:]
    return encoder.decode(truncated)

class DynamicContextManager:
    def __init__(self, model: str = "gpt-4o", reserved_output_tokens: int = 2000):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_context = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
        }.get(model, 128000)
        self.reserved_output = reserved_output_tokens

    def fit_messages(self, messages: list[dict], system_prompt: str = "") -> list[dict]:
        system_tokens = len(self.encoder.encode(system_prompt)) if system_prompt else 0
        available = self.max_context - self.reserved_output - system_tokens
        fitted = []
        for msg in reversed(messages):
            msg_tokens = len(self.encoder.encode(msg["content"]))
            if available - msg_tokens >= 0:
                fitted.insert(0, msg)
                available -= msg_tokens
            else:
                break
        if system_prompt:
            fitted.insert(0, {"role": "system", "content": system_prompt})
        return fitted

    def compress_history(self, conversation: list[dict], target_ratio: float = 0.5) -> list[dict]:
        summaries = []
        token_count = sum(len(self.encoder.encode(m["content"])) for m in conversation)
        if token_count <= self.max_context * target_ratio:
            return conversation
        compressed = []
        for msg in conversation:
            tokens = self.encoder.encode(msg["content"])
            if len(tokens) > 500:
                summary = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Compress this to under 50 words, preserving key info:\n\n{msg['content'][:2000]}"
                    }],
                    temperature=0, max_tokens=100
                ).choices[0].message.content
                compressed.append({"role": msg["role"], "content": summary})
            else:
                compressed.append(msg)
        return compressed
```

---

## 3. Batching & Parallelization

### Request Batching

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    def __init__(self, model: str = "gpt-4o-mini", max_workers: int = 10):
        self.model = model
        self.max_workers = max_workers

    def process_batch(self, items: list[str], prompt_template: str, system_prompt: str = "") -> list[str]:
        results = [None] * len(items)

        def process_single(i: int, item: str) -> tuple[int, str]:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_template.format(input=item)}
                ],
                temperature=0
            )
            return i, response.choices[0].message.content

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(process_single, i, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

        return results

    def process_batch_with_fallback(self, items: list[str], prompt_template: str,
                                     system_prompt: str = "", fallback_model: str = "gpt-3.5-turbo") -> list[str]:
        results = []
        for item in items:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_template.format(input=item)}
                    ],
                    temperature=0,
                    timeout=10
                )
                results.append(response.choices[0].message.content)
            except Exception:
                fallback = openai.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": prompt_template.format(input=item)}],
                    temperature=0
                )
                results.append(fallback.choices[0].message.content)
        return results
```

### Async Processing

```python
import asyncio
import aiohttp

class AsyncLLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_concurrent: int = 20):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def chat_completion(self, messages: list[dict], **kwargs) -> str:
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        **kwargs
                    }
                ) as resp:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

    async def process_many(self, prompts: list[str], system_prompt: str = "") -> list[str]:
        tasks = [
            self.chat_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p}
            ])
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    client = AsyncLLMClient(api_key="sk-...", max_concurrent=20)
    prompts = [f"Tell me about {topic}" for topic in ["Python", "Rust", "Go", "Java"]]
    results = await client.process_many(prompts, "Answer concisely.")
    for topic, result in zip(prompts, results):
        if isinstance(result, Exception):
            print(f"Failed: {result}")
        else:
            print(f"{topic}: {result[:50]}...")
```

---

## 4. Model Routing

Route requests to the appropriate model based on complexity, cost constraints, and latency requirements.

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "cheap_fast": {
                "model": "gpt-4o-mini",
                "cost_per_1k_input": 0.00015,
                "cost_per_1k_output": 0.0006,
                "latency_p50_ms": 300,
                "suitable_for": ["classification", "extraction", "simple_qa", "summarization"]
            },
            "balanced": {
                "model": "gpt-4o",
                "cost_per_1k_input": 0.0025,
                "cost_per_1k_output": 0.01,
                "latency_p50_ms": 800,
                "suitable_for": ["reasoning", "code", "complex_qa", "analysis"]
            },
            "reasoning": {
                "model": "o1-preview",
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.06,
                "latency_p50_ms": 5000,
                "suitable_for": ["math", "logic", "planning", "research"]
            }
        }

    def route(self, task_type: str, complexity: str = "low",
              max_latency_ms: int = 2000, max_cost: float = 0.01) -> str:
        candidates = []
        for name, config in self.models.items():
            if task_type in config["suitable_for"]:
                if config["latency_p50_ms"] <= max_latency_ms:
                    candidates.append((name, config))

        if not candidates:
            cheapest = min(self.models.items(), key=lambda x: x[1]["cost_per_1k_input"])
            return cheapest[0]

        if complexity == "low" and max_cost < 0.001:
            return "cheap_fast"
        elif complexity == "high" or task_type in ("math", "logic", "planning"):
            return "reasoning"
        else:
            return "balanced"

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        config = self.models[model_name]
        return (
            (input_tokens / 1000) * config["cost_per_1k_input"] +
            (output_tokens / 1000) * config["cost_per_1k_output"]
        )

class SmartRouter:
    def __init__(self):
        self.router = ModelRouter()
        self.fallback_chain = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    async def execute(self, messages: list[dict], task_type: str,
                      complexity: str = "low", **kwargs) -> dict:
        model_name = self.router.route(task_type, complexity)
        models_to_try = [model_name] + [
            m for m in self.fallback_chain if m != model_name
        ]

        for model in models_to_try:
            try:
                start = time.time()
                response = openai.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                latency_ms = (time.time() - start) * 1000

                usage = response.usage
                cost = self.router.estimate_cost(
                    model, usage.prompt_tokens, usage.completion_tokens
                )

                return {
                    "content": response.choices[0].message.content,
                    "model_used": model,
                    "latency_ms": round(latency_ms, 2),
                    "cost": round(cost, 6),
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "fallback_used": model != model_name
                }
            except Exception as e:
                continue

        return {"error": "All models failed", "content": None}

    async def execute_with_cascade(self, prompt: str, system_prompt: str = "",
                                    min_quality: float = 0.8) -> dict:
        """
        Start with cheapest model, evaluate quality, escalate if needed.
        """
        configs = [
            ("gpt-4o-mini", 0),
            ("gpt-4o", 0),
        ]

        for model, temperature in configs:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            content = response.choices[0].message.content

            quality = await self._evaluate_quality(prompt, content)
            if quality >= min_quality:
                return {
                    "content": content,
                    "model_used": model,
                    "quality_score": quality
                }

        return {
            "content": content,
            "model_used": configs[-1][0],
            "quality_score": quality,
            "note": "Used best model, quality still below threshold"
        }

    async def _evaluate_quality(self, prompt: str, response: str) -> float:
        eval_resp = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Rate response quality 0-1:\n\nQ: {prompt}\nA: {response}\n\nScore:"
            }],
            temperature=0,
            max_tokens=5
        )
        try:
            return float(eval_resp.choices[0].message.content.strip())
        except ValueError:
            return 0.5
```

---

## 5. Latency Optimization

### Streaming with Early Exit

```python
def stream_with_early_exit(messages: list[dict], stop_condition: callable,
                            model: str = "gpt-4o-mini") -> str:
    """
    Stream response and stop early when a condition is met.
    stop_condition: called with partial text, returns True to stop.
    """
    collected = []
    stream = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            collected.append(delta)
            partial = "".join(collected)
            if stop_condition(partial):
                break

    return "".join(collected)

def classification_with_early_exit(text: str, categories: list[str]) -> str:
    """Classify text and stop as soon as a category token appears."""
    messages = [{
        "role": "user",
        "content": f"Classify into exactly one: {categories}\n\nText: {text}\n\nCategory:"
    }]
    return stream_with_early_exit(
        messages,
        stop_condition=lambda partial: any(cat in partial for cat in categories)
    )
```

### Connection Pooling & Keep-Alive

```python
import httpx

class OptimizedHTTPClient:
    def __init__(self):
        self.client = httpx.Client(
            base_url="https://api.openai.com",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30
            )
        )

    def chat_completion(self, model: str, messages: list[dict], **kwargs) -> dict:
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, **kwargs}
        )
        return response.json()

http_client = OptimizedHTTPClient()

# Reuse the same connection for all calls
for _ in range(100):
    result = http_client.chat_completion(
        "gpt-4o-mini",
        [{"role": "user", "content": "Say hello"}]
    )
```

---

## 6. Cost Optimization Strategies

### Tiered Prompt Strategy

```python
class CostOptimizer:
    def __init__(self):
        self.cheap_model = "gpt-4o-mini"
        self.expensive_model = "gpt-4o"
        self.token_estimator = tiktoken.encoding_for_model("gpt-4o")

    def should_use_cheap_model(self, prompt: str, task_type: str) -> bool:
        prompt_tokens = len(self.token_estimator.encode(prompt))
        if prompt_tokens > 5000:
            return False
        cheap_tasks = {"classification", "extraction", "simple_qa", "summarization"}
        return task_type in cheap_tasks

    def optimize_prompt_length(self, prompt: str, max_tokens: int = 2000) -> str:
        tokens = self.token_estimator.encode(prompt)
        if len(tokens) <= max_tokens:
            return prompt
        return self.token_estimator.decode(tokens[:max_tokens])

    def minimize_output_tokens(self, messages: list[dict], task_type: str) -> int:
        output_budgets = {
            "classification": 10,
            "extraction": 100,
            "summarization": 300,
            "qa": 500,
            "generation": 1000,
            "analysis": 2000
        }
        return output_budgets.get(task_type, 500)

    def execute(self, messages: list[dict], task_type: str) -> dict:
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]

        model = self.cheap_model if self.should_use_cheap_model(user, task_type) else self.expensive_model
        max_tokens = self.minimize_output_tokens(messages, task_type)

        start = time.time()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
        latency_ms = (time.time() - start) * 1000

        cost = (
            (response.usage.prompt_tokens / 1000) * (0.00015 if model == "gpt-4o-mini" else 0.0025) +
            (response.usage.completion_tokens / 1000) * (0.0006 if model == "gpt-4o-mini" else 0.01)
        )

        return {
            "content": response.choices[0].message.content,
            "model": model,
            "cost": round(cost, 6),
            "savings": cost < 0.001,
            "latency_ms": round(latency_ms, 2)
        }
```

### Batch API (OpenAI Batch API)

```python
class BatchAPIProcessor:
    """
    Use OpenAI's Batch API for 50% cost reduction on async workloads.
    Batch responses have a 24-hour SLA but cost half.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def prepare_batch_file(self, requests: list[dict]) -> str:
        lines = []
        for i, req in enumerate(requests):
            line = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req["model"],
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0),
                    "max_tokens": req.get("max_tokens", 500)
                }
            }
            lines.append(json.dumps(line))

        content = "\n".join(lines)
        file_path = f"/tmp/batch_{int(time.time())}.jsonl"
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def submit_batch(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            upload = openai.files.create(file=f, purpose="batch")

        batch = openai.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    def retrieve_results(self, batch_id: str) -> list[dict]:
        import time
        while True:
            batch = openai.batches.retrieve(batch_id)
            if batch.status == "completed":
                break
            elif batch.status == "failed":
                raise Exception(f"Batch failed: {batch.errors}")
            time.sleep(30)

        result = openai.files.content(batch.output_file_id)
        results = []
        for line in result.text.strip().split("\n"):
            results.append(json.loads(line))
        return results
```

---

## 7. Observability & Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("prompt-engineering")

def traced_llm_call(prompt_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(messages: list[dict], **kwargs):
            with tracer.start_as_current_span(
                f"llm.{prompt_name}",
                kind=SpanKind.CLIENT
            ) as span:
                span.set_attribute("prompt.name", prompt_name)
                span.set_attribute("prompt.model", kwargs.get("model", "gpt-4o-mini"))
                span.set_attribute("prompt.temperature", kwargs.get("temperature", 0))
                span.set_attribute("prompt.max_tokens", kwargs.get("max_tokens", 500))
                span.set_attribute("prompt.input_length", len(str(messages)))

                start = time.time()
                try:
                    result = func(messages, **kwargs)
                    duration = (time.time() - start) * 1000
                    span.set_attribute("prompt.duration_ms", duration)
                    span.set_attribute("prompt.output_length", len(str(result)))
                    span.set_attribute("prompt.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("prompt.success", False)
                    span.set_attribute("prompt.error", str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator

@traced_llm_call("classifier")
def classify(messages: list[dict], **kwargs):
    return openai.chat.completions.create(**kwargs)
```

### Structured Logging with Request IDs

```python
import uuid
import structlog

logger = structlog.get_logger()

class LLMRequestTracker:
    def __init__(self):
        self.request_id = str(uuid.uuid4())
        self.start_time = None
        self.messages = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000
        log_data = {
            "request_id": self.request_id,
            "duration_ms": round(duration, 2),
            "success": exc_type is None
        }
        if exc_val:
            log_data["error"] = str(exc_val)
        logger.info("llm_request", **log_data)

    async def track(self, coro):
        self.start_time = time.time()
        try:
            result = await coro
            duration = (time.time() - self.start_time) * 1000
            logger.info("llm_request_completed",
                request_id=self.request_id,
                duration_ms=round(duration, 2),
                success=True
            )
            return result
        except Exception as e:
            duration = (time.time() - self.start_time) * 1000
            logger.error("llm_request_failed",
                request_id=self.request_id,
                duration_ms=round(duration, 2),
                error=str(e)
            )
            raise

async def tracked_request():
    tracker = LLMRequestTracker()
    async with tracker:
        result = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        return result
```

---

## 8. Security & Rate Limit Management

### Exponential Backoff with Jitter

```python
import random

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

class RetryHandler:
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute(self, func: callable, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                delay = exponential_backoff(attempt, self.base_delay)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                last_error = e
            except openai.APITimeoutError as e:
                delay = exponential_backoff(attempt, self.base_delay * 0.5)
                logger.warning(f"Timeout, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                last_error = e
            except (openai.APIConnectionError, httpx.ConnectError) as e:
                delay = exponential_backoff(attempt, self.base_delay)
                logger.warning(f"Connection error, retrying in {delay:.1f}s")
                time.sleep(delay)
                last_error = e
        raise last_error
```

---

## 9. Architecture Patterns

### Producer-Consumer with Queue

```python
import asyncio
from asyncio import Queue

class LLMQueueProcessor:
    def __init__(self, num_workers: int = 5, queue_size: int = 100):
        self.queue = Queue(maxsize=queue_size)
        self.num_workers = num_workers
        self.workers = []

    async def producer(self, requests: list[dict]):
        for req in requests:
            await self.queue.put(req)
        for _ in range(self.num_workers):
            await self.queue.put(None)

    async def worker(self, worker_id: int):
        while True:
            request = await self.queue.get()
            if request is None:
                self.queue.task_done()
                break
            try:
                response = await openai.chat.completions.create(**request)
                await self.handle_result(worker_id, request, response)
            except Exception as e:
                await self.handle_error(worker_id, request, e)
            finally:
                self.queue.task_done()

    async def handle_result(self, worker_id: int, request: dict, response):
        logger.info(f"Worker {worker_id} completed: {response.usage.total_tokens} tokens")

    async def handle_error(self, worker_id: int, request: dict, error: Exception):
        logger.error(f"Worker {worker_id} failed: {error}")

    async def run(self, requests: list[dict]):
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        await self.producer(requests)
        await self.queue.join()
        for w in workers:
            w.cancel()
```

### Health Check & Circuit Breaker Dashboard

```python
class SystemHealthMonitor:
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "errors": 0,
            "total_latency_ms": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.recent_latencies = deque(maxlen=100)
        self.error_timestamps = deque(maxlen=100)

    def record_call(self, latency_ms: float, cost: float, success: bool, cached: bool = False):
        self.metrics["total_calls"] += 1
        self.metrics["total_latency_ms"] += latency_ms
        self.metrics["total_cost"] += cost
        self.recent_latencies.append(latency_ms)
        if cached:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
        if not success:
            self.metrics["errors"] += 1
            self.error_timestamps.append(time.time())

    def get_health_status(self) -> dict:
        total = self.metrics["total_calls"]
        if total == 0:
            return {"status": "no_data"}
        avg_latency = self.metrics["total_latency_ms"] / total
        error_rate = self.metrics["errors"] / total
        cache_rate = self.metrics["cache_hits"] / total if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0

        recent_errors = sum(
            1 for t in self.error_timestamps
            if time.time() - t < 300
        )

        status = "healthy"
        if error_rate > 0.1 or recent_errors > 10:
            status = "degraded"
        if error_rate > 0.25:
            status = "critical"

        return {
            "status": status,
            "total_calls": total,
            "avg_latency_ms": round(avg_latency, 1),
            "p95_latency_ms": round(sorted(self.recent_latencies)[
                int(len(self.recent_latencies) * 0.95)
            ], 1) if len(self.recent_latencies) >= 20 else None,
            "error_rate": round(error_rate, 4),
            "cache_hit_rate": round(cache_rate, 4),
            "total_cost_usd": round(self.metrics["total_cost"], 4),
            "recent_errors_5min": recent_errors,
            "timestamp": datetime.now().isoformat()
        }
```

---

## 10. Optimization Decision Framework

| Goal | Primary Strategy | Secondary Strategy | Trade-off |
|------|-----------------|-------------------|-----------|
| Reduce cost | Exact-match caching | Model routing (cheaper model) | Cache misses still cost full price |
| Reduce latency | Streaming + early exit | Connection pooling | Higher complexity |
| Improve throughput | Batching + async | Producer-consumer queue | More memory usage |
| Improve reliability | Retries + fallback models | Circuit breaker | Higher latency under failures |
| Handle scale | Tiered caching | Queue-based processing | Infrastructure cost |
| Reduce token usage | Prompt compression | Dynamic context management | Potential quality loss |
| Cut API costs 50% | Batch API (24h window) | Prompt optimization | Delayed results |
| Optimize for quality | Cascade routing (cheap → expensive) | Self-consistency voting | Higher cost per query |
