# Prompting Techniques

This document catalogs the major prompting techniques used to elicit reliable, high-quality outputs from LLMs. Each technique includes its rationale, implementation patterns, and code examples.

---

## 1. Zero-Shot Prompting

**Zero-shot prompting** asks the model to perform a task without providing any examples. The model relies entirely on its pre-trained knowledge and instruction-following ability.

### When to Use
- Simple, well-defined tasks (classification, translation, summarization).
- Baseline before adding examples.
- Cost-sensitive applications (fewer tokens = lower cost).

### Pattern

```
<instruction>
<input>
```

### Code Example

```python
def zero_shot_classify(text: str, categories: list[str]) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify the following text into one of these categories: {', '.join(categories)}.\n\nText: {text}\n\nCategory:"
        }],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

print(zero_shot_classify(
    "Tesla's stock dropped 5% today due to production concerns.",
    ["Technology", "Finance", "Healthcare", "Education"]
))
# Output: Finance
```

### Limitations
- Poor performance on nuanced or domain-specific tasks.
- Sensitive to wording and phrasing.
- May produce inconsistent formatting.

---

## 2. Few-Shot Prompting

**Few-shot prompting** provides 2-5 examples of input-output pairs before asking the model to complete the task. This technique, first highlighted in the GPT-3 paper, dramatically improves performance on structured tasks.

### Pattern

```
Task description.

Example 1:
Input: <input1>
Output: <output1>

Example 2:
Input: <input2>
Output: <output2>

Input: <target_input>
Output:
```

### Code Example

```python
def few_shot_extract(text: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": """Extract product name, price, and availability from product descriptions.

Example 1:
Input: "The new MacBook Pro 16-inch with M3 Max chip is available for $3499. Ships within 2 weeks."
Output: {"product": "MacBook Pro 16-inch M3 Max", "price": 3499, "available": true}

Example 2:
Input: "Sony WH-1000XM5 headphones - currently out of stock. Regular price $399.99."
Output: {"product": "Sony WH-1000XM5 headphones", "price": 399.99, "available": false}

Input: """ + text + """
Output:"""
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

data = few_shot_extract("iPad Air 11-inch M2: $599, in stock now.")
print(data)
# {"product": "iPad Air 11-inch M2", "price": 599, "available": true}
```

### Selecting Examples

Use retrieval-augmented example selection (dynamic few-shot):

```python
import numpy as np

def select_examples(query: str, example_pool: list[dict], k: int = 3) -> list[dict]:
    """Select the most similar examples using embedding similarity."""
    query_emb = openai.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding
    
    similarities = []
    for ex in example_pool:
        ex_emb = openai.embeddings.create(
            model="text-embedding-3-small", input=ex["input"]
        ).data[0].embedding
        sim = np.dot(query_emb, ex_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(ex_emb)
        )
        similarities.append((sim, ex))
    
    similarities.sort(reverse=True)
    return [ex for _, ex in similarities[:k]]
```

---

## 3. Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** prompting, introduced by Wei et al. (2022), instructs the model to show its reasoning step-by-step before giving the final answer. This technique significantly improves performance on arithmetic, logic, and multi-step reasoning tasks.

### Pattern

Add "Let's think step by step" or provide a reasoning example.

### Code Example

```python
def solve_with_cot(problem: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Solve this problem step by step.\n\nProblem: {problem}\n\nLet's think step by step:"
        }],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content

result = solve_with_cot(
    "A store has 120 apples. It sells 40% of them in the morning and "
    "half of the remaining in the afternoon. How many apples are left?"
)
print(result)
# Step-by-step reasoning...
# Final answer: 36 apples
```

### Zero-Shot CoT

Simply appending `"Let's think step by step"` works surprisingly well:

```python
def zero_shot_cot(question: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Let's think step by step:"}
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content
```

### Structured CoT (Output Parsing)

For production, separate reasoning from answer:

```python
def structured_cot(question: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Solve the following problem.

First, reason step by step inside <reasoning> tags.
Then, provide the final answer inside <answer> tags.

Problem: {question}"""
        }],
        temperature=0,
        max_tokens=500
    )
    
    content = response.choices[0].message.content
    
    # Parse structured output
    import re
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    
    return {
        "reasoning": reasoning.group(1).strip() if reasoning else "",
        "answer": answer.group(1).strip() if answer else ""
    }

result = structured_cot("If x = 5 and y = 3, what is 2x + 3y?")
print(result["reasoning"])
print(f"Answer: {result['answer']}")  # 19
```

---

## 4. Tree-of-Thought (ToT) Prompting

**Tree-of-Thought** (Yao et al., 2023) extends CoT by exploring multiple reasoning branches simultaneously. The model evaluates intermediate states and prunes unpromising paths, similar to a search algorithm.

### When to Use

- Complex planning tasks.
- Mathematical proofs.
- Creative problem-solving where multiple approaches are viable.

### Implementation Pattern

```python
def tree_of_thought(problem: str, branches: int = 3, depth: int = 2) -> list[dict]:
    """Simple ToT implementation using iterative prompting."""
    current_level = [{"path": [], "context": problem}]
    
    for level in range(depth):
        next_level = []
        
        for node in current_level:
            prompt = f"""Problem: {problem}

Current reasoning: {' '.join(node['path'])}

Generate {branches} distinct next steps. For each step:
- Rate confidence (1-10)
- Explain why this path is promising

Format each step as:
Step: <description>
Confidence: <score>
Rationale: <explanation>"""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300
            )
            
            # Parse branches (simplified)
            content = response.choices[0].message.content
            steps = parse_steps(content)  # Custom parser
            
            for step in steps:
                next_level.append({
                    "path": node["path"] + [step["description"]],
                    "confidence": step["confidence"],
                    "rationale": step["rationale"]
                })
        
        # Prune: keep top branches by confidence
        next_level.sort(key=lambda x: x["confidence"], reverse=True)
        current_level = next_level[:branches]
    
    return current_level
```

---

## 5. ReAct Prompting

**ReAct** (Reasoning + Acting, Yao et al., 2023) interleaves reasoning traces with actions (tool calls, API queries). The model thinks, acts, observes, and iterates.

### Pattern

```
Thought: <reasoning>
Action: <tool_name>
Action Input: <tool_input>
Observation: <tool_output>
... (repeat)
Thought: <final_reasoning>
Final Answer: <answer>
```

### Code Example

```python
import json

def react_agent(question: str, tools: dict, max_steps: int = 5) -> str:
    """
    tools: dict of {"tool_name": callable}
    """
    system = """You are a ReAct agent. You have access to the following tools:
- search: Search the web. Input: a query string.
- calculate: Perform arithmetic. Input: a math expression.

Use this format:

Thought: your reasoning
Action: tool_name
Action Input: tool_input
Observation: tool_result
... (repeat as needed)
Thought: I now have the answer
Final Answer: your_answer"""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    
    for step in range(max_steps):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        
        if "Final Answer:" in content:
            return content.split("Final Answer:")[-1].strip()
        
        # Parse action
        if "Action:" in content and "Action Input:" in content:
            action = content.split("Action:")[1].split("\n")[0].strip()
            action_input = content.split("Action Input:")[1].split("\n")[0].strip()
            
            if action in tools:
                try:
                    observation = tools[action](action_input)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = f"Unknown tool: {action}"
            
            messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return "Failed to reach a final answer."

# Example tools
def search(query: str) -> str:
    return f"Results for '{query}': Paris is the capital of France."

def calculate(expr: str) -> str:
    return str(eval(expr, {"__builtins__": {}}, {}))

tools = {"search": search, "calculate": calculate}

answer = react_agent(
    "What is the population of the capital of France, divided by 1000?",
    tools
)
print(answer)
```

---

## 6. Self-Consistency

**Self-Consistency** (Wang et al., 2022) runs the same prompt multiple times (with temperature > 0) and takes the majority answer. It improves reliability by averaging across reasoning paths.

### Code Example

```python
from collections import Counter

def self_consistency(prompt: str, num_samples: int = 5, model: str = "gpt-4o") -> str:
    responses = []
    for i in range(num_samples):
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Non-zero for diversity
            max_tokens=200
        )
        responses.append(response.choices[0].message.content)
        print(f"Sample {i+1}: {responses[-1]}")
    
    # Extract and vote on final answers
    # Assumes answers are the last line or follow "Answer:"
    answers = []
    for r in responses:
        if "Answer:" in r:
            answers.append(r.split("Answer:")[-1].strip())
        else:
            answers.append(r.strip().split("\n")[-1])
    
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    print(f"\nConsensus: {most_common} (agreement: {counter[most_common]}/{num_samples})")
    return most_common

result = self_consistency(
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    num_samples=5
)
```

---

## 7. Role Prompting / Persona Prompting

Assigning the model a specific **role** or **persona** before giving the task. This leverages the model's training data about how different roles communicate.

### Patterns

```python
def role_prompt(task: str, role: str, style_notes: str = "") -> str:
    system = f"You are an expert {role}. {style_notes}"
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Different personas for the same task
task = "Explain what DNS is."

print("=== Teacher ===")
print(role_prompt(task, "teacher", "Explain to a 10-year-old. Use analogies."))

print("\n=== Network Engineer ===")
print(role_prompt(task, "network engineer", "Use technical terminology. Be precise."))

print("\n=== Product Manager ===")
print(role_prompt(task, "product manager", "Focus on business value and user impact."))
```

---

## 8. Structured Outputs (JSON Mode / Grammar)

Ensuring the model returns parseable structured data. Modern models support native JSON mode, and local models support grammar-constrained generation.

### OpenAI JSON Mode

```python
def extract_structured(text: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Extract entities from this text. Return valid JSON.\n\n{text}"
        }],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

print(extract_structured(
    "Apple Inc. was founded by Steve Jobs in Cupertino, CA on April 1, 1976."
))
```

### Pydantic Integration

```python
from pydantic import BaseModel, Field
from typing import Optional

class ExtractedEntity(BaseModel):
    organization: Optional[str] = None
    founder: Optional[str] = None
    location: Optional[str] = None
    founded_date: Optional[str] = None

def extract_with_schema(text: str, schema: BaseModel) -> dict:
    response = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Extract information:\n\n{text}"}],
        response_format=schema
    )
    return response.choices[0].message.parsed

result = extract_with_schema(
    "Apple Inc. was founded by Steve Jobs in Cupertino, CA on April 1, 1976.",
    ExtractedEntity
)
print(result.model_dump())
```

---

## 9. RAG Prompting Patterns

**Retrieval-Augmented Generation (RAG)** augments prompts with relevant context retrieved from a knowledge base. The prompt design for RAG is critical.

### Basic RAG Prompt

```python
def rag_prompt(query: str, context_docs: list[str]) -> str:
    context = "\n\n".join([
        f"[Document {i+1}]: {doc}" for i, doc in enumerate(context_docs)
    ])
    
    prompt = f"""Answer the question based only on the provided context. If the context does not contain enough information, say "I don't have enough information to answer."

Context:
{context}

Question: {query}

Answer:"""
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content

docs = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "Python 3.0 was released in 2008."
]
print(rag_prompt("Who created Python?", docs))
```

### Advanced: Citation-Enforced RAG

```python
def rag_with_citations(query: str, context_docs: list[dict]) -> dict:
    """
    context_docs: [{"id": "doc1", "content": "...", "source": "..."}]
    """
    context_str = "\n\n".join([
        f"[{doc['id']}] {doc['content']}"
        for doc in context_docs
    ])
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Answer the question using the provided context. 
Cite sources using their IDs in brackets. If multiple sources support a claim, cite all of them.
If the context is insufficient, state that clearly.

Context:
{context_str}

Question: {query}

Answer:"""
        }],
        temperature=0,
        max_tokens=500
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources_used": list(set(
            f"[{doc['id']}]"
            for doc in context_docs
        ))
    }
```

---

## 10. Prompt Chaining

Breaking a complex task into multiple sequential prompts, where each prompt's output feeds into the next. This modular approach improves reliability and debuggability.

### Example: Document Analysis Pipeline

```python
class PromptChain:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    def call(self, system: str, prompt: str, temperature: float = 0) -> str:
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def analyze_document(self, document: str) -> dict:
        # Step 1: Summarize
        summary = self.call(
            "You are a document analyst.",
            f"Summarize this document in 2-3 sentences:\n\n{document}"
        )
        
        # Step 2: Extract key entities
        entities = self.call(
            "You are an entity extraction system.",
            f"Extract all named entities (people, organizations, dates, locations) as a JSON list.\n\n{document}",
            response_format={"type": "json_object"}
        )
        
        # Step 3: Sentiment analysis
        sentiment = self.call(
            "You are a sentiment analyst.",
            f"Classify the overall sentiment of this document as positive, negative, or neutral.\n\n{document}"
        )
        
        # Step 4: Generate recommendations
        recommendations = self.call(
            "You are a strategic consultant.",
            f"Based on this document, provide 3 actionable recommendations.\n\nDocument: {document}\nSummary: {summary}"
        )
        
        return {
            "summary": summary,
            "entities": json.loads(entities) if isinstance(entities, str) else entities,
            "sentiment": sentiment.strip(),
            "recommendations": recommendations
        }

chain = PromptChain()
result = chain.analyze_document("Quarterly earnings report...")
```

---

## 11. Constitutional AI / Self-Critique

The model evaluates and revises its own output against a set of rules (the "constitution"). This improves safety and quality without external classifiers.

```python
def self_critique(text: str, rules: list[str]) -> tuple[str, list[str]]:
    """Generate text, critique it, revise it."""
    
    # Initial generation
    initial = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}],
        temperature=0.7
    ).choices[0].message.content
    
    # Critique
    critique_prompt = f"""Review the following text against these rules:
{chr(10).join(f'- {r}' for r in rules)}

Text:
{initial}

Identify any violations and explain how to fix them."""
    
    critique = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0
    ).choices[0].message.content
    
    # Revise
    revision_prompt = f"""Original text:
{initial}

Feedback:
{critique}

Please revise the original text to address the feedback while preserving the core message."""
    
    revised = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    ).choices[0].message.content
    
    return revised, [initial, critique]

rules = [
    "Do not make exaggerated claims or hyperbole.",
    "Be specific and cite numbers when possible.",
    "Avoid overly promotional language."
]

final, history = self_critique("Write a product description for our new laptop.", rules)
```

---

## Technique Selection Guide

| Task Type | Recommended Technique | Why |
|-----------|----------------------|-----|
| Classification | Zero-shot or Few-shot | Simple, fast, cheap |
| Structured extraction | Few-shot + JSON mode | Reliable parsing |
| Math / Logic | CoT + Self-Consistency | Step-by-step + voting |
| Planning | Tree-of-Thought | Branch exploration |
| Multi-step reasoning with tools | ReAct | Interleaved thinking & actions |
| Long-form content | Prompt chaining | Modular, debuggable |
| Open-ended QA | RAG | Grounded in knowledge |
| Safety-critical | Constitutional AI | Self-regulation |
| Any task in production | Structured outputs | Reliable parsing |
