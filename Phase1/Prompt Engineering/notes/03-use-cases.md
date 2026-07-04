# Prompt Engineering Use Cases

This document catalogs real-world applications of prompt engineering across domains, with implementation patterns and code examples for each.

---

## 1. Code Generation & Review

### Use Case: Generating Unit Tests

```python
def generate_unit_tests(function_code: str, framework: str = "pytest") -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"""You are a senior software engineer. Generate comprehensive {framework} unit tests.
Cover: happy path, edge cases, error conditions.
Use descriptive test names and docstrings."""
        }, {
            "role": "user",
            "content": f"Generate tests for:\n\n{function_code}"
        }],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

code = """
def calculate_discount(price: float, discount_percent: float) -> float:
    if price < 0:
        raise ValueError("Price cannot be negative")
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""

print(generate_unit_tests(code))
```

### Use Case: Code Review

```python
def review_code(code: str, language: str = "python") -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"""Review this {language} code for:
1. Bugs and logic errors
2. Performance issues
3. Security vulnerabilities
4. Style and maintainability
5. Missing error handling

Rate each category: ✅ Pass, ⚠️ Warning, ❌ Fail"""
        }, {
            "role": "user",
            "content": code
        }],
        temperature=0,
        max_tokens=1000
    )
    
    # Parse structured output
    content = response.choices[0].message.content
    return parse_review(content)  # Custom parser

review = review_code("""
def fetch_data(url):
    import requests
    r = requests.get(url)
    return r.json()
""")
```

### Use Case: Code Translation

```python
def translate_code(source_code: str, from_lang: str, to_lang: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Translate this {from_lang} code to {to_lang}. Preserve logic, naming, and comments:\n\n{source_code}"
        }],
        temperature=0,
        max_tokens=2000
    )
    return response.choices[0].message.content

python_code = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
"""

print(translate_code(python_code, "python", "rust"))
```

---

## 2. Text Summarization

### Use Case: Multi-Level Summarization

```python
def summarize_at_level(text: str, level: str = "short") -> str:
    level_config = {
        "tltr": (1, 30, "one sentence, maximum 30 characters"),
        "short": (2, 100, "2-3 sentences, maximum 100 words"),
        "medium": (4, 300, "a paragraph of 4-5 sentences, max 300 words"),
        "detailed": (8, 800, "comprehensive with key details, max 800 words"),
        "bullet": (0, 200, "3-5 bullet points covering main points")
    }
    
    sentences, words, instruction = level_config[level]
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize the following text in {instruction}:\n\n{text}"
        }],
        temperature=0.3,
        max_tokens=words * 2
    )
    return response.choices[0].message.content

article = "..."  # Long article
print(summarize_at_level(article, "tltr"))
print(summarize_at_level(article, "bullet"))
```

### Use Case: Meeting Minutes

```python
def generate_meeting_minutes(transcript: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Extract structured meeting minutes from this transcript:

{transcript}

Return a JSON with:
- date (if mentioned)
- attendees (list)
- key_discussions (list of topics)
- decisions_made (list)
- action_items (list of {task, owner, deadline})
- next_steps (list)"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

---

## 3. Data Extraction & Classification

### Use Case: Entity Extraction Pipeline

```python
class EntityExtractor:
    def __init__(self):
        self.schema = {
            "people": [{"name": "", "role": "", "mentions": 0}],
            "organizations": [{"name": "", "type": ""}],
            "locations": [],
            "dates": [],
            "metrics": [{"metric": "", "value": "", "unit": ""}]
        }
    
    def extract(self, text: str) -> dict:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Extract all entities from this text and return as JSON:

{text}

Schema: {json.dumps(self.schema, indent=2)}"""
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    
    def batch_extract(self, texts: list[str]) -> list[dict]:
        """Process multiple texts concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.extract, texts))
        return results

extractor = EntityExtractor()
result = extractor.extract(
    "Apple announced Q4 2024 revenue of $89.5 billion. Tim Cook said "
    "growth was driven by Services. The new office in Bangalore will open in March."
)
```

### Use Case: Multi-Label Text Classification

```python
def classify_document(text: str, categories: list[str], 
                      multi_label: bool = False) -> dict:
    prompt = f"""Classify the following text{' into one category' if not multi_label else ', assigning all that apply'} from this list:
{categories}

Text: {text}

Classification:"""
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    
    result = response.choices[0].message.content.strip()
    return {
        "text": text[:100],
        "categories": result.split(", ") if multi_label else [result]
    }

# Single-label
print(classify_document(
    "Stock markets rallied on Fed rate cut expectations.",
    ["Finance", "Technology", "Healthcare", "Politics"]
))

# Multi-label
print(classify_document(
    "Apple's new AI-powered health monitoring watch gets FDA approval.",
    ["Technology", "Healthcare", "Finance", "Regulatory"],
    multi_label=True
))
```

---

## 4. Conversational Agents

### Use Case: Customer Support Agent

```python
class SupportAgent:
    def __init__(self, knowledge_base: dict):
        self.kb = knowledge_base
        self.conversation_history = []
    
    def _build_context(self) -> str:
        return json.dumps(self.kb, indent=2)
    
    def respond(self, user_message: str) -> str:
        self.conversation_history.append({
            "role": "user", "content": user_message
        })
        
        system = f"""You are a helpful customer support agent for AcmeCorp.

Knowledge Base:
{self._build_context()}

Guidelines:
- Be empathetic and professional
- If you don't know something, say so honestly
- Always ask clarifying questions when needed
- Provide step-by-step instructions for technical issues
- Escalate to human agent if the issue is unresolved after 3 exchanges"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                *self.conversation_history[-10:]  # Keep last 10 messages
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        reply = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant", "content": reply
        })
        return reply

kb = {
    "return_policy": "30-day return window, original packaging required",
    "shipping": "Free shipping on orders over $50, standard 5-7 business days",
    "contact": "support@acmecorp.com or 1-800-555-0199, hours 9AM-6PM EST"
}

agent = SupportAgent(kb)
print(agent.respond("I want to return my laptop, what do I do?"))
```

### Use Case: Intent Router

```python
def route_intent(user_message: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify this user message into one intent:

Intents:
- order_status: Checking order delivery status
- return_request: Initiating a return/refund
- technical_support: Getting help with a product issue
- billing: Payment or invoice questions
- general_inquiry: Product info, pricing, availability
- complaint: Negative feedback or escalation
- off_topic: Not related to our business

Respond with JSON: {{"intent": "<intent>", "confidence": <0-1>, "entities": {{...}}}}

Message: {user_message}"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

print(route_intent("Where is my order #12345?"))
# {"intent": "order_status", "confidence": 0.98, "entities": {"order_id": "12345"}}
```

---

## 5. Automated Reasoning & Planning

### Use Case: Task Decomposition

```python
def decompose_task(task: str, max_subtasks: int = 8) -> list[dict]:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Decompose this task into {max_subtasks} or fewer subtasks.

For each subtask, provide:
- id: number
- description: what to do
- dependencies: list of subtask IDs that must be done first
- estimated_effort: "low", "medium", "high"
- output: what this subtask produces

Return as JSON.

Task: {task}"""
        }],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)

plan = decompose_task("Build a REST API for a todo list application with user authentication")
for subtask in plan["subtasks"]:
    print(f"{subtask['id']}. {subtask['description']} [{subtask['estimated_effort']}]")
```

### Use Case: Decision Making with Pros/Cons

```python
def analyze_decision(decision: str, criteria: list[str]) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Analyze this decision using a structured framework.

Decision: {decision}
Evaluation criteria: {criteria}

Provide:
1. Pros (with evidence or reasoning)
2. Cons (with evidence or reasoning)
3. Risk assessment (probability × impact)
4. Recommended option
5. Contingency plan if things go wrong

Return as JSON."""
        }],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)

result = analyze_decision(
    "Should we migrate our monolith to microservices?",
    ["Development cost", "Team expertise", "Timeline", "Scalability needs", "Operational complexity"]
)
```

---

## 6. Content Moderation

### Use Case: Multi-Aspect Moderation

```python
def moderate_content(text: str) -> dict:
    categories = [
        "hate_speech", "harassment", "self_harm", "sexual_content",
        "violence", "misinformation", "spam", "copyright_violation"
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Review this content against our moderation policies.

Categories to check: {', '.join(categories)}

For each category, return:
- flagged: true/false
- severity: "none", "low", "medium", "high", "critical"
- reason: brief explanation
- policy_cited: which policy applies

Return as JSON.

Content: {text}"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    result["action_required"] = any(
        c["severity"] in ("high", "critical") 
        for c in result.get("categories", {}).values()
    )
    return result

# Integrated with OpenAI's built-in moderation
def hybrid_moderation(text: str) -> dict:
    # Layer 1: Built-in API
    mod_response = openai.moderations.create(input=text)
    api_flags = mod_response.results[0].categories
    
    # Layer 2: LLM-based nuanced review
    llm_review = moderate_content(text)
    
    return {
        "api_flagged": any(
            getattr(api_flags, cat) for cat in vars(api_flags)
        ),
        "llm_review": llm_review,
        "overall_verdict": "rejected" if (
            any(getattr(api_flags, cat) for cat in vars(api_flags))
            or llm_review.get("action_required")
        ) else "approved"
    }
```

---

## 7. Data Augmentation & Synthetic Data

### Use Case: Generate Training Data

```python
def generate_training_examples(
    task_description: str,
    num_examples: int = 10,
    output_format: str = "json"
) -> list[dict]:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Generate {num_examples} diverse training examples for the following task.

Task: {task_description}

Each example should include:
- "input": the input/prompt
- "expected_output": the correct output
- "difficulty": "easy", "medium", "hard"
- "explanation": why this output is correct

Make examples varied and cover edge cases.
Return as a JSON array.

Output format: {output_format}"""
        }],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=3000
    )
    return json.loads(response.choices[0].message.content)

samples = generate_training_examples(
    "Given a customer review, classify sentiment as positive, negative, or neutral, "
    "and extract the key pain points or highlights.",
    num_examples=5
)
```

---

## 8. Translation & Localization

### Use Case: Context-Aware Translation

```python
def translate_with_context(text: str, target_lang: str, 
                           context: str, tone: str = "formal") -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Translate the following text to {target_lang}.

Context: This text is from a {context}.
Tone: {tone}

Requirements:
- Preserve meaning and nuance
- Adapt idioms appropriately (don't translate literally)
- Maintain the original format (paragraphs, line breaks)
- Use culturally appropriate expressions

Text to translate:
{text}

Translation:"""
        }],
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content

print(translate_with_context(
    "It's raining cats and dogs outside.",
    "Spanish",
    "casual conversation between friends",
    "informal"
))
```

---

## 9. Question Answering

### Use Case: Multi-Hop QA

```python
def answer_multi_hop(question: str, context: str) -> dict:
    """Answer questions requiring multiple inference steps."""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Answer the question by breaking it into sub-questions.

Context:
{context}

Question: {question}

Format your response as JSON:
{{
  "sub_questions": ["sub-q1", "sub-q2", ...],
  "intermediate_answers": ["ans1", "ans2", ...],
  "final_answer": "the complete answer",
  "confidence": 0.0-1.0,
  "supporting_evidence": ["quote1", "quote2"]
}}"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

context = """
The Eiffel Tower was built in 1889 for the World's Fair in Paris.
It is 330 meters tall. Marie Curie won the Nobel Prize in Physics in 1903.
She was born in Warsaw, Poland in 1867. The Eiffel Tower was the world's 
tallest structure until the Chrysler Building was built in 1930.
"""

print(answer_multi_hop("What was the world's tallest structure in 1900?", context))
```

---

## 10. Prompt Optimization (Meta-Prompting)

### Use Case: Auto-Prompt Improvement

```python
def optimize_prompt(task_description: str, initial_prompt: str,
                    test_cases: list[dict]) -> str:
    """
    Automatically improve a prompt based on test case performance.
    test_cases: [{"input": "...", "expected": "..."}]
    """
    
    # Evaluate current prompt
    results = []
    for tc in test_cases:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": initial_prompt},
                      {"role": "user", "content": tc["input"]}],
            temperature=0
        )
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": response.choices[0].message.content
        })
    
    # Ask the model to improve the prompt
    improvement_prompt = f"""Task: {task_description}

Current prompt:
{initial_prompt}

Test results (input → expected → actual):
{json.dumps(results, indent=2)}

Analyze the failures and suggest an improved prompt that would fix them.
Focus on:
1. Clarity of instructions
2. Format specification
3. Edge case handling
4. Constraint enforcement

Provide the improved prompt:"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": improvement_prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content
```

---

## Industry-Specific Use Cases

| Industry | Use Case | Key Technique |
|----------|----------|---------------|
| Healthcare | Clinical note summarization | RAG + CoT |
| Legal | Contract clause extraction | Few-shot + JSON mode |
| Finance | Earnings report analysis | Prompt chaining |
| E-commerce | Product description generation | Role prompting |
| Education | Adaptive tutoring | Conversational agent |
| Media | Content personalization | RAG + classification |
| DevOps | Incident root cause analysis | ReAct (tool use) |
| HR | Resume screening | Structured extraction |
