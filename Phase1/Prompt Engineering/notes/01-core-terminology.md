# Core Terminology in Prompt Engineering

Prompt engineering is the discipline of designing, refining, and optimizing inputs to large language models (LLMs) to produce desired outputs. This document covers the foundational vocabulary every practitioner must know.

---

## 1. Prompt

A **prompt** is the input text provided to an LLM that instructs it on what to generate. Prompts can range from a single sentence to multi-paragraph structured instructions. The quality and structure of a prompt directly determine the quality of the model's response.

### Types of Prompts

- **System Prompt**: A directive given to the model at the start of a conversation to set behavior, tone, and constraints. System prompts are often hidden from end users in chat interfaces.
- **User Prompt**: The input from the end user. In chat-based models, this is the message sent after the system prompt.
- **Assistant Prompt / Prefix**: Sometimes used to steer the model's response by providing a partial completion (e.g., `"The answer is"` to force a specific format).
- **Meta-Prompt**: A prompt that instructs the model on how to construct another prompt (common in auto-prompt generation systems).

### Example

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
# Output: "Arrr, the capital of France be Paris, me hearty!"
```

---

## 2. Token

**Tokens** are the atomic units of text that an LLM processes. They are not individual characters or words, but rather subword units determined by a tokenizer. For example, the word "unhappiness" might be split into `["un", "happiness"]` or `["unh", "appi", "ness"]` depending on the tokenizer.

### Key Facts

- **1 token ≈ 0.75 words** for English text (roughly 4 characters per token).
- Models have a **maximum context window** measured in tokens (e.g., 8k, 32k, 128k, or 1M for some models).
- Pricing is typically per token (e.g., $0.01 per 1K input tokens).
- Tokenizers are model-specific — the same text produces different token counts across models.

### Code Example: Counting Tokens

```python
import tiktoken  # OpenAI's tokenizer library

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)

text = "Prompt engineering is the art of communicating with LLMs."
print(f"Token count: {count_tokens(text)}")
# Output: Token count: 10
```

### Tokens and Cost Estimation

```python
def estimate_cost(input_text: str, output_text: str, model: str = "gpt-4o") -> dict:
    encoder = tiktoken.encoding_for_model(model)
    input_tokens = len(encoder.encode(input_text))
    output_tokens = len(encoder.encode(output_text))
    
    # Approximate pricing (check latest pricing from provider)
    input_price_per_1k = 0.0025
    output_price_per_1k = 0.01
    
    input_cost = (input_tokens / 1000) * input_price_per_1k
    output_cost = (output_tokens / 1000) * output_price_per_1k
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 6)
    }

prompt = "Explain quantum computing in simple terms."
response = "Quantum computing uses qubits that can be in multiple states at once."
print(estimate_cost(prompt, response))
```

---

## 3. Context Window

The **context window** is the total number of tokens a model can process in a single request — combining both input (prompt) and output (completion). It is the model's "working memory."

### Implications

- If the prompt exceeds the context window, it must be truncated, which may lose critical information.
- Long-context models (e.g., Gemini 1.5 Pro with 1M tokens, GPT-4-128k) enable processing entire books, codebases, or hour-long transcripts.
- Larger context windows come with higher computational cost and latency.

### Managing Context Windows

```python
def fit_within_context(messages: list, system_prompt: str, 
                       max_context: int = 128000, model: str = "gpt-4o") -> list:
    """Truncate conversation history to fit within context window."""
    encoder = tiktoken.encoding_for_model(model)
    available = max_context - len(encoder.encode(system_prompt)) - 500  # reserve for response
    
    fitted = []
    # Process messages in reverse (keep newest)
    for msg in reversed(messages):
        msg_tokens = len(encoder.encode(msg["content"]))
        if available - msg_tokens >= 0:
            fitted.insert(0, msg)
            available -= msg_tokens
        else:
            break
    
    return fitted

history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "..." * 50000},  # Very long response
    {"role": "user", "content": "Tell me more."}
]

safe_history = fit_within_context(history, "You are a helpful assistant.")
print(f"Kept {len(safe_history)} of {len(history)} messages")
```

---

## 4. Temperature

**Temperature** controls the randomness of the model's output. It is a hyperparameter that scales the logits (raw scores) before the softmax probability distribution is computed.

### Behavior

- **Temperature = 0**: Deterministic — the model always picks the most likely token. Use for factual tasks, classification, code generation.
- **Temperature ≈ 0.7**: Balanced — some creativity while maintaining coherence. Use for general conversation.
- **Temperature ≈ 1.0**: Highly random — diverse but potentially nonsensical outputs. Use for creative writing, brainstorming.
- **Temperature > 1.0**: Very high entropy — can produce incoherent text. Rarely used.

### Code Example

```python
import openai

def generate_with_temperature(prompt: str, temperature: float) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100
    )
    return response.choices[0].message.content

prompt = "Write a one-sentence tagline for a lemonade stand."

print("Temp 0.0:", generate_with_temperature(prompt, 0.0))
print("Temp 0.7:", generate_with_temperature(prompt, 0.7))
print("Temp 1.5:", generate_with_temperature(prompt, 1.5))
```

---

## 5. Top-p (Nucleus Sampling)

**Top-p**, also called nucleus sampling, is an alternative to temperature-based sampling. Instead of considering all possible next tokens, the model samples only from the smallest set of tokens whose cumulative probability exceeds `p`.

### Behavior

- **Top-p = 0.1**: Only the top 10% probability mass is considered — very focused.
- **Top-p = 0.9**: 90% of probability mass is considered — more diverse.
- **Top-p = 1.0**: All tokens are considered (effectively disabled).

### Relationship with Temperature

Temperature and Top-p can be used together:
- Temperature flattens or sharpens the probability distribution.
- Top-p then determines how much of that distribution to sample from.

A common recommendation: **adjust temperature OR top-p, not both aggressively**. Start with temperature tuning, then use top-p for fine-grained control.

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate a recipe for a fruit salad."}],
    temperature=0.8,
    top_p=0.9
)
```

---

## 6. Top-k

**Top-k** sampling restricts the model to consider only the `k` most likely next tokens. The probability distribution is re-normalized over these `k` tokens before sampling.

### Behavior

- **Top-k = 1**: Greedy decoding (same as temperature = 0).
- **Top-k = 40**: Default for many models — reasonable diversity.
- **Top-k = 100+**: Very diverse output.

### When to Use Top-k

Top-k is useful when you want to prevent the model from ever choosing very unlikely tokens. It is less commonly used than top-p in modern practice but can be effective for:
- Constrained generation (e.g., valid JSON tokens only).
- Ensuring output stays within a known vocabulary subset.

---

## 7. Logit Bias

**Logit bias** allows you to increase or decrease the probability of specific tokens appearing in the output. You provide a bias value (ranging from -100 to +100) for specific token IDs.

### Use Cases

- Forcing the model to output in a specific format (e.g., always start with "Answer:").
- Blocking disallowed words or phrases.
- Steering sentiment (e.g., making outputs more positive).

### Code Example

```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
# Find token IDs for the word "excellent"
token_id = encoder.encode("excellent")[0]
print(f"Token ID for 'excellent': {token_id}")

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Rate this product: it works okay."}],
    logit_bias={str(token_id): 10},  # Boost probability of "excellent"
    max_tokens=50
)
print(response.choices[0].message.content)
```

---

## 8. Frequency Penalty & Presence Penalty

These parameters discourage the model from repeating itself.

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Frequency Penalty** | -2.0 to 2.0 | Penalizes tokens that have already appeared, proportional to their frequency. Higher values → more diverse vocabulary. |
| **Presence Penalty** | -2.0 to 2.0 | Penalizes tokens that have already appeared at all, regardless of frequency. Higher values → more topic switching. |

### Practical Guidance

- **Frequency penalty** for tasks where you want lexical diversity (e.g., creative writing, synonym generation).
- **Presence penalty** for brainstorming or exploring multiple perspectives.
- Negative values encourage repetition (useful for maintaining consistent formatting).

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 10 ways to improve sleep quality."}],
    frequency_penalty=0.5,
    presence_penalty=0.3
)
```

---

## 9. Stop Sequences

**Stop sequences** are strings that, when generated by the model, immediately halt further generation. They are critical for controlling output boundaries.

### Use Cases

- Ensuring the model doesn't generate beyond a single response in a structured format.
- Truncating at a delimiter (e.g., `\n\n`, `---`, `</output>`).
- Parsing multi-part responses (e.g., stop at `NEXT` to handle interleaved reasoning).

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 planets:\n1."}],
    stop=["\n\n", "---"],
    max_tokens=200
)
```

---

## 10. Max Tokens

**Max tokens** limits the length of the model's response. This is a hard cutoff — the model stops generating once this limit is reached.

### Considerations

- Always reserve tokens for the output when managing context windows.
- Short max tokens → faster responses, lower cost.
- Long max tokens → more comprehensive but higher cost and latency.

```python
# Reserve output budget
MAX_CONTEXT = 128000
OUTPUT_BUDGET = 4000
INPUT_BUDGET = MAX_CONTEXT - OUTPUT_BUDGET

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a detailed report..."}],
    max_tokens=OUTPUT_BUDGET
)
```

---

## 11. Embedding

An **embedding** is a dense vector representation of text in a high-dimensional space. Embeddings capture semantic meaning — texts with similar meanings have vectors that are close together (by cosine similarity).

### Use Cases

- Semantic search / retrieval (RAG).
- Clustering and topic modeling.
- Classification (train a classifier on top of embeddings).
- Recommendation systems.

### Code Example

```python
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="Prompt engineering is fascinating."
)

embedding = response.data[0].embedding
print(f"Dimension: {len(embedding)}")  # 1536 for text-embedding-3-small
print(f"First 5 values: {embedding[:5]}")
```

### Similarity Computation

```python
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

texts = ["I love programming", "Coding is my passion", "I enjoy eating pizza"]
embeddings = [
    openai.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
    for t in texts
]

similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity between text 1 and 2: {similarity:.3f}")  # High (0.85+)
```

---

## 12. Completion

A **completion** is the output generated by the model in response to a prompt. In chat-based models, completions are structured as assistant messages that can include both content and tool calls.

### Structured Completions

Modern models support structured outputs via JSON mode:

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract the name and age from: John is 28 years old."}],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
print(data)  # {"name": "John", "age": 28}
```

---

## 13. Logprobs

**Logprobs** (log-probabilities) provide the log probability of each generated token. This is useful for:
- Measuring model confidence in its output.
- Detecting hallucinations (low confidence in factual claims).
- Building custom sampling strategies.

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    logprobs=True,
    top_logprobs=3
)

choice = response.choices[0]
for token, logprob_info in zip(choice.message.content.split(), choice.logprobs.content):
    print(f"Token: {token}")
    for top in logprob_info.top_logprobs[:3]:
        print(f"  {top.token}: {np.exp(top.logprob):.3f}")
```

---

## 14. System Prompt vs. User Prompt Separation

Modern chat-based models distinguish between system and user messages:

```python
SYSTEM_PROMPT = """You are a code review assistant. Follow these rules:
1. Always start with a summary of changes.
2. List issues in priority order.
3. Suggest fixes with code examples.
4. Keep tone constructive and professional."""

USER_PROMPT = """Review this Python function:
def add(a,b):
    return a+b"""

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
)
```

**Best practice**: Use the system prompt for persistent instructions and the user prompt for the specific task. This separation improves reliability and maintainability.

---

## Summary Table

| Term | Purpose | Typical Values | Affects |
|------|---------|----------------|---------|
| Prompt | Input instruction | N/A | Everything |
| Token | Atomic unit of text | N/A | Cost, context limits |
| Context Window | Total processing capacity | 4K - 1M tokens | What can be included |
| Temperature | Output randomness | 0.0 - 1.0 (rarely >1) | Creativity vs. determinism |
| Top-p | Nucleus sampling threshold | 0.1 - 1.0 | Output diversity |
| Top-k | Top-k token restriction | 1 - 100+ | Output diversity |
| Logit Bias | Token probability adjustment | -100 to +100 | Token-level behavior |
| Frequency Penalty | Repetition penalty (frequency) | -2.0 to 2.0 | Vocabulary diversity |
| Presence Penalty | Repetition penalty (existence) | -2.0 to 2.0 | Topic diversity |
| Stop Sequences | Generation halt markers | Strings | Output boundaries |
| Max Tokens | Output length limit | 1 - context_size | Response length, cost |
| Embedding | Semantic vector representation | 256 - 3072 dims | Search, classification |
| Logprobs | Token probability log | Negative floats | Confidence, debugging |
