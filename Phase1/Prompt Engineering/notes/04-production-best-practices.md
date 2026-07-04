# Production Best Practices for Prompt Engineering

**Production Note #1: Reliability, Safety, and Maintenance**

Moving prompt engineering from notebooks to production requires systematic thinking about versioning, error handling, security, and cost management. This note covers the infrastructure and practices needed to run prompts reliably at scale.

---

## 1. Prompt Versioning & Management

Treat prompts as first-class code artifacts. They should be versioned, reviewed, and deployed through the same pipelines as application code.

### Prompt as Code

Store prompts in a dedicated directory with a structured format:

```python
# prompts/summarization/v1.yaml
name: article_summarizer
version: "1.2.0"
model: gpt-4o-mini
temperature: 0.3
max_tokens: 500
system: |
  You are a professional editor. Summarize the following article in 2-3 sentences.
  Focus on: key findings, data points, and conclusions.
  Maintain a neutral tone.
user_template: |
  Article: {{article_text}}
  
  Summary:
parameters:
  article_text: string
```

### Prompt Registry

```python
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class PromptConfig(BaseModel):
    name: str
    version: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 500
    system: str
    user_template: str
    parameters: dict[str, str]
    tags: list[str] = []
    description: Optional[str] = None

class PromptRegistry:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptConfig] = {}
        self._load_all()
    
    def _load_all(self):
        for yaml_file in self.prompts_dir.rglob("*.yaml"):
            with open(yaml_file) as f:
                config = PromptConfig(**yaml.safe_load(f))
                key = f"{config.name}:{config.version}"
                self._cache[key] = config
    
    def get(self, name: str, version: str = "latest") -> PromptConfig:
        if version == "latest":
            # Find highest version for this name
            candidates = {
                k: v for k, v in self._cache.items() 
                if k.startswith(f"{name}:")
            }
            if not candidates:
                raise KeyError(f"Prompt '{name}' not found")
            version = max(candidates.keys(), key=lambda k: k.split(":")[1])
        return self._cache[f"{name}:{version}"]
    
    def render(self, name: str, params: dict, version: str = "latest") -> dict:
        config = self.get(name, version)
        
        # Validate parameters
        for key, param_type in config.parameters.items():
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        
        from jinja2 import Template
        user_prompt = Template(config.user_template).render(**params)
        
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.system},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }

# Usage
registry = PromptRegistry()
request = registry.render("article_summarizer", {
    "article_text": "Long article content here..."
})
response = openai.chat.completions.create(**request)
```

### Semantic Versioning for Prompts

- **MAJOR**: Breaking changes (different output format, different model family).
- **MINOR**: New features (added examples, new parameter).
- **PATCH**: Bug fixes (fixed typo, adjusted temperature).

---

## 2. Guardrails & Content Filtering

Production systems need multiple layers of protection to handle both input attacks and output risks.

### Input Validation Layer

```python
class InputGuard:
    """Validate and sanitize user inputs before sending to the LLM."""
    
    MAX_INPUT_LENGTH = 32000  # characters
    
    def __init__(self):
        self.blocked_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)",
            r"system\s+prompt",
            r"you\s+are\s+(now|not)",
            r"<[^>]*script[^>]*>",  # XSS attempts
        ]
        self.token_limit = 8000
    
    def validate(self, user_input: str) -> tuple[bool, str]:
        # Length check
        if len(user_input) > self.MAX_INPUT_LENGTH:
            return False, "Input exceeds maximum length"
        
        # Injection patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, "Input contains prohibited patterns"
        
        # Token estimation (rough: 4 chars per token)
        estimated_tokens = len(user_input) // 4
        if estimated_tokens > self.token_limit:
            return False, f"Input too long (~{estimated_tokens} tokens)"
        
        return True, "ok"
    
    def sanitize(self, text: str) -> str:
        """Strip known dangerous patterns."""
        text = re.sub(r"system:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"assistant:\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

guard = InputGuard()
is_valid, message = guard.validate(user_input)
if not is_valid:
    return {"error": message}, 400
```

### Output Guard Layer

```python
class OutputGuard:
    """Check LLM outputs before returning to the user."""
    
    def __init__(self):
        self.sensitive_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card number (basic)
            r"[\w\.-]+@[\w\.-]+\.\w+",  # Email (if PII redaction needed)
        ]
        self.min_output_length = 1
        self.max_output_length = 32000
    
    def check(self, output: str, user_input: str = "") -> dict:
        issues = []
        
        # Length bounds
        if len(output) < self.min_output_length:
            issues.append("Empty or too short response")
        if len(output) > self.max_output_length:
            issues.append("Response exceeds maximum length")
        
        # Sensitive data leakage
        for pattern in self.sensitive_patterns:
            if re.search(pattern, output):
                issues.append("Potential PII leakage detected")
        
        # Repetition detection
        if self._is_repetitive(output):
            issues.append("Response contains excessive repetition")
        
        # Refusal detection (model refusing to answer when it shouldn't)
        refusal_phrases = [
            "I cannot answer", "I'm unable to", "I am unable to",
            "As an AI", "I don't have enough information"
        ]
        # Only flag if the refusal seems unjustified
        # (This is context-dependent and may need a secondary LLM check)
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "truncated": len(output) > self.max_output_length
        }
    
    def _is_repetitive(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text contains too many repeated n-grams."""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check bigram repetition
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique = len(set(bigrams))
        ratio = 1 - (unique / len(bigrams))
        return ratio > threshold

output_guard = OutputGuard()
```

### Content Moderation Pipeline

```python
class ModerationPipeline:
    """Multi-layer content safety system."""
    
    def __init__(self):
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
    
    def process(self, user_input: str, llm_callable: callable) -> dict:
        # Layer 1: Input validation
        valid, msg = self.input_guard.validate(user_input)
        if not valid:
            return {"status": "rejected", "reason": msg, "output": None}
        
        # Layer 2: OpenAI moderation API
        mod = openai.moderations.create(input=user_input)
        if mod.results[0].flagged:
            return {
                "status": "rejected",
                "reason": "Input flagged by content moderation",
                "categories": mod.results[0].categories,
                "output": None
            }
        
        # Layer 3: Call LLM
        try:
            llm_output = llm_callable(user_input)
        except Exception as e:
            return {"status": "error", "reason": str(e), "output": None}
        
        # Layer 4: Output guard
        check = self.output_guard.check(llm_output, user_input)
        if not check["passed"]:
            return {
                "status": "blocked",
                "reason": f"Output guard triggered: {check['issues']}",
                "output": llm_output  # Log original but don't return
            }
        
        # Layer 5: Output moderation
        out_mod = openai.moderations.create(input=llm_output)
        if out_mod.results[0].flagged:
            return {
                "status": "blocked",
                "reason": "Output flagged by content moderation",
                "output": llm_output
            }
        
        return {"status": "passed", "output": llm_output}
```

---

## 3. Error Handling & Fallbacks

Production LLM calls must handle failures gracefully. Implement a layered fallback strategy.

```python
class LLMClient:
    """Resilient LLM client with retries, fallbacks, and error handling."""
    
    def __init__(self):
        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]  # Ordered by preference
        self.max_retries = 3
        self.timeout = 30
    
    def generate(self, messages: list[dict], **kwargs) -> str:
        last_error = None
        
        for model in self.models:
            for attempt in range(self.max_retries):
                try:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=self.timeout,
                        **kwargs
                    )
                    return response.choices[0].message.content
                
                except openai.RateLimitError as e:
                    wait = min(2 ** attempt * 2, 30)  # Exponential backoff
                    logger.warning(f"Rate limited on {model}, waiting {wait}s")
                    time.sleep(wait)
                    last_error = e
                
                except openai.APITimeoutError as e:
                    logger.warning(f"Timeout on {model}, attempt {attempt}")
                    last_error = e
                
                except openai.APIConnectionError as e:
                    logger.warning(f"Connection error on {model}, attempt {attempt}")
                    time.sleep(1)
                    last_error = e
                
                except openai.BadRequestError as e:
                    # Don't retry client errors
                    raise
                
                except Exception as e:
                    logger.error(f"Unexpected error on {model}: {e}")
                    last_error = e
            
            # Move to next model
            logger.info(f"Falling back from {model}")
        
        raise Exception(f"All models failed. Last error: {last_error}")
    
    def generate_with_fallback_response(self, messages: list[dict], 
                                         fallback: str = "I'm sorry, I encountered an error. Please try again.",
                                         **kwargs) -> str:
        """Generate with a user-facing fallback string."""
        try:
            return self.generate(messages, **kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return fallback

    def generate_structured(self, messages: list[dict], 
                            response_schema: type, **kwargs) -> dict:
        """Generate structured output with validation."""
        for attempt in range(self.max_retries):
            try:
                response = openai.beta.chat.completions.parse(
                    model=self.models[0],
                    messages=messages,
                    response_format=response_schema,
                    **kwargs
                )
                data = response.choices[0].message.parsed
                return data.model_dump()
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "partial": None}
                time.sleep(1)

client = LLMClient()
result = client.generate_with_fallback_response(messages)
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevent cascading failures by halting calls after repeated errors."""
    
    def __init__(self, threshold: int = 5, recovery_timeout: int = 60):
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: callable, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.threshold:
                self.state = "open"
            raise

class CircuitBreakerOpen(Exception):
    pass
```

---

## 4. Rate Limiting & Concurrency

### Token Bucket Rate Limiter

```python
import time
import asyncio
from collections import deque

class TokenBucket:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
    
    def acquire(self, tokens: int = 1) -> float:
        """Returns wait time in seconds. 0 means immediate."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            else:
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
                self.tokens = 0  # Drain
                return wait_time

class AsyncRateLimiter:
    """Async-aware rate limiter with queue."""
    
    def __init__(self, rpm: int = 60, concurrency: int = 10):
        self.rpm = rpm
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.request_times: deque = deque()
    
    async def acquire(self):
        async with self.semaphore:
            now = time.monotonic()
            # Clean old requests (older than 1 minute)
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()
            
            # Check if we can make another request
            if len(self.request_times) >= self.rpm:
                sleep_time = self.request_times[0] + 60 - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.monotonic())
            return True

# Usage
rate_limiter = AsyncRateLimiter(rpm=60, concurrency=10)

async def rate_limited_call(messages: list[dict]) -> str:
    await rate_limiter.acquire()
    return await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
```

### Request Batching

```python
def batch_process(items: list, batch_fn: callable, batch_size: int = 20) -> list:
    """Process items in batches with progress tracking."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch (parallel if batch_fn supports it)
        batch_results = batch_fn(batch)
        results.extend(batch_results)
        
        # Progress
        progress = min(i + batch_size, len(items))
        logger.info(f"Processed {progress}/{len(items)} items")
    
    return results

def batch_classify(texts: list[str], categories: list[str]) -> list[dict]:
    """Batch classification using a single API call per batch."""
    batch_prompt = "\n\n---\n\n".join([
        f"Text {i+1}: {t}" for i, t in enumerate(texts)
    ])
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify each text into one of: {', '.join(categories)}.
Return a JSON object mapping text numbers to categories.

{batch_prompt}"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## 5. Cost Tracking & Budget Management

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class LLMCallRecord:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    prompt_name: str
    user_id: str
    success: bool
    cost: float

class CostTracker:
    # Updated pricing (per 1K tokens) - check provider for latest
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    }
    
    def __init__(self):
        self.records: list[LLMCallRecord] = []
        self.daily_budget: dict[str, float] = {}  # date -> budget
    
    def record(self, response, model: str, prompt_name: str, 
               user_id: str, latency_ms: int, success: bool = True):
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )
        
        record = LLMCallRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            prompt_name=prompt_name,
            user_id=user_id,
            success=success,
            cost=cost
        )
        self.records.append(record)
        self._check_budget(cost)
        return record
    
    def _check_budget(self, cost: float):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_budget:
            self.daily_budget[today] = 0
        self.daily_budget[today] += cost
        
        daily_limit = 100.0  # $100/day
        if self.daily_budget[today] > daily_limit:
            logger.warning(f"Daily budget exceeded! Cost so far: ${self.daily_budget[today]:.2f}")
            # Trigger alert
    
    def get_stats(self, hours: int = 24) -> dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.records if r.timestamp > cutoff]
        
        return {
            "total_calls": len(recent),
            "total_cost": sum(r.cost for r in recent),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in recent),
            "avg_latency_ms": sum(r.latency_ms for r in recent) / len(recent) if recent else 0,
            "success_rate": sum(1 for r in recent if r.success) / len(recent) if recent else 0,
            "by_model": {
                model: sum(r.cost for r in recent if r.model == model)
                for model in set(r.model for r in recent)
            },
            "by_prompt": {
                name: sum(r.cost for r in recent if r.prompt_name == name)
                for name in set(r.prompt_name for r in recent)
            }
        }

# Usage
tracker = CostTracker()
response = openai.chat.completions.create(model="gpt-4o-mini", messages=msgs)
tracker.record(response, "gpt-4o-mini", "classifier", "user_123", latency_ms=450)
```

---

## 6. Logging & Observability

```python
import structlog
from datetime import datetime

logger = structlog.get_logger()

class LLMLogger:
    """Structured logging for all LLM interactions."""
    
    def log_interaction(self, prompt_name: str, messages: list[dict],
                        response: str, metadata: dict, duration_ms: float):
        logger.info("llm_interaction",
            prompt_name=prompt_name,
            model=metadata.get("model"),
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            duration_ms=round(duration_ms, 2),
            cost=metadata.get("cost"),
            user_id=metadata.get("user_id"),
            session_id=metadata.get("session_id"),
            timestamp=datetime.utcnow().isoformat(),
            # Don't log full content for PII reasons
            input_length=len(str(messages)),
            output_length=len(response),
            success=metadata.get("success", True),
            error=metadata.get("error")
        )
    
    def log_error(self, prompt_name: str, error: Exception, 
                  messages: list[dict], metadata: dict):
        logger.error("llm_error",
            prompt_name=prompt_name,
            error_type=type(error).__name__,
            error_message=str(error),
            model=metadata.get("model"),
            user_id=metadata.get("user_id"),
            timestamp=datetime.utcnow().isoformat()
        )

class PromptAnalytics:
    """Tracks prompt performance metrics over time."""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client  # Optional: for real-time metrics
    
    def record(self, prompt_name: str, latency_ms: float, 
               success: bool, tokens_used: int, cost: float):
        metrics = {
            "latency_p50": self._percentile("latency", prompt_name, 50),
            "latency_p99": self._percentile("latency", prompt_name, 99),
            "success_rate": self._rolling_success_rate(prompt_name),
            "avg_cost": self._average_cost(prompt_name),
            "total_calls": self._total_calls(prompt_name),
        }
        return metrics
    
    def _percentile(self, metric: str, prompt: str, p: int) -> float:
        # Implement with database query
        pass
    
    def _rolling_success_rate(self, prompt: str, window: int = 100) -> float:
        # Last N calls success rate
        pass

# Logging middleware decorator
def logged_llm_call(prompt_name: str):
    def decorator(func: callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"llm_call_start", prompt_name=prompt_name)
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.info(f"llm_call_end", 
                    prompt_name=prompt_name,
                    duration_ms=round(duration, 2),
                    success=True
                )
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(f"llm_call_error",
                    prompt_name=prompt_name,
                    duration_ms=round(duration, 2),
                    error=str(e)
                )
                raise
        return wrapper
    return decorator
```

---

## 7. A/B Testing Prompts in Production

```python
import random

class PromptExperiment:
    """A/B test different prompt versions in production."""
    
    def __init__(self, name: str, variants: list[dict], 
                 traffic_split: list[float] = None):
        """
        variants: [{"name": "control", "config": {...}}, {"name": "test", "config": {...}}]
        traffic_split: [0.5, 0.5] for equal split
        """
        self.name = name
        self.variants = variants
        self.traffic_split = traffic_split or [1.0 / len(variants)] * len(variants)
        self.results = {v["name"]: [] for v in variants}
    
    def get_variant(self, user_id: str = None) -> dict:
        """Deterministic or random assignment."""
        if user_id:
            # Consistent assignment for the same user
            seed = hash(f"{self.name}:{user_id}")
            random.seed(seed)
        
        r = random.random()
        cumulative = 0
        for i, split in enumerate(self.traffic_split):
            cumulative += split
            if r <= cumulative:
                return self.variants[i]
        
        return self.variants[-1]
    
    def record_result(self, variant_name: str, latency_ms: float, 
                      success: bool, quality_score: float = None):
        self.results[variant_name].append({
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "success": success,
            "quality_score": quality_score
        })
    
    def analyze(self) -> dict:
        report = {}
        for name, records in self.results.items():
            if not records:
                report[name] = {"error": "No data"}
                continue
            
            successes = [r for r in records if r.success]
            scores = [r["quality_score"] for r in records if r["quality_score"] is not None]
            
            report[name] = {
                "calls": len(records),
                "success_rate": len(successes) / len(records),
                "avg_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
                "avg_quality": sum(scores) / len(scores) if scores else None
            }
        
        return report

# Production usage
experiment = PromptExperiment(
    name="summarization_v2",
    variants=[
        {"name": "v1_baseline", "system": "Summarize concisely."},
        {"name": "v2_structured", "system": "Summarize with bullet points."}
    ],
    traffic_split=[0.5, 0.5]
)

def handle_request(user_id: str, text: str):
    variant = experiment.get_variant(user_id)
    start = time.time()
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": variant["system"]},
                {"role": "user", "content": text}
            ]
        )
        success = True
        quality = assess_quality(response.choices[0].message.content)  # Custom scorer
    except Exception:
        success = False
        quality = 0
    
    latency = (time.time() - start) * 1000
    experiment.record_result(variant["name"], latency, success, quality)
    
    return response
```

---

## 8. Prompt Testing Suite

```python
class PromptTestCase:
    def __init__(self, input: str, expected: str = None,
                 expected_contains: list[str] = None,
                 expected_json_schema: dict = None,
                 expected_min_length: int = None):
        self.input = input
        self.expected = expected
        self.expected_contains = expected_contains or []
        self.expected_json_schema = expected_json_schema
        self.expected_min_length = expected_min_length

class PromptTester:
    def __init__(self, render_fn: callable):
        """
        render_fn: function that takes input and returns (messages, kwargs)
        """
        self.render_fn = render_fn
    
    def run_test(self, test_case: PromptTestCase) -> dict:
        try:
            messages, kwargs = self.render_fn(test_case.input)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=kwargs.get("max_tokens", 500)
            )
            output = response.choices[0].message.content
            
            results = {"passed": True, "failures": []}
            
            if test_case.expected and output.strip() != test_case.expected.strip():
                results["failures"].append(f"Expected '{test_case.expected}', got '{output.strip()}'")
            
            for substring in test_case.expected_contains:
                if substring not in output:
                    results["failures"].append(f"Missing expected substring: '{substring}'")
            
            if test_case.expected_json_schema:
                try:
                    data = json.loads(output)
                    # Validate against schema (simplified)
                    for key in test_case.expected_json_schema:
                        if key not in data:
                            results["failures"].append(f"Missing JSON key: '{key}'")
                except json.JSONDecodeError:
                    results["failures"].append("Output is not valid JSON")
            
            if test_case.expected_min_length and len(output) < test_case.expected_min_length:
                results["failures"].append(
                    f"Output too short: {len(output)} < {test_case.expected_min_length}"
                )
            
            results["passed"] = len(results["failures"]) == 0
            results["output"] = output
            return results
            
        except Exception as e:
            return {"passed": False, "failures": [str(e)], "output": None}
    
    def run_suite(self, test_cases: list[PromptTestCase]) -> dict:
        results = []
        for i, tc in enumerate(test_cases):
            result = self.run_test(tc)
            result["test_id"] = i
            results.append(result)
        
        passed = sum(1 for r in results if r["passed"])
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "details": results
        }
```

---

## 9. Deployment Checklist

Before deploying a prompt to production, verify:

- [ ] **Prompt versioned** — stored in registry, has semantic version.
- [ ] **Input validation** — length, injection patterns, type checking.
- [ ] **Output validation** — format, quality, safety checks.
- [ ] **Error handling** — retries, fallback models, fallback responses.
- [ ] **Rate limiting** — per-user and global limits configured.
- [ ] **Cost tracking** — per-prompt costs monitored, budget alerts set.
- [ ] **Logging** — all calls logged with metadata (no PII in logs).
- [ ] **A/B test** — variants defined, traffic split configured.
- [ ] **Tests** — at minimum 5-10 test cases covering happy path, edge cases, errors.
- [ ] **Monitoring** — latency, success rate, cost dashboards set up.
- [ ] **Human review** — for high-stakes prompts, reviewed by a second engineer.
- [ ] **Documentation** — prompt purpose, parameters, expected behavior documented.
