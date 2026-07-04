# Evaluation & Iteration for Prompt Engineering

**Production Note #2: Measuring Quality and Continuously Improving Prompts**

Without systematic evaluation, prompt engineering is just guesswork. This note covers quantitative and qualitative methods for measuring prompt quality, building evaluation pipelines, and creating feedback loops for continuous improvement.

---

## 1. Why Evaluation Matters

In production, a prompt that works "most of the time" is not good enough. Small prompt changes can have outsized effects:

- A single word change can flip sentiment classification.
- Removing a few-shot example can drop accuracy by 20%.
- Temperature adjustments can make outputs unreliable.

Without evaluation, you cannot:
- Detect regressions when changing prompts.
- Compare prompt variants objectively.
- Know when your prompt is "good enough" to deploy.
- Debug why a prompt fails on specific inputs.

---

## 2. Evaluation Strategies

### Human Evaluation

The gold standard, but expensive and slow. Use for:
- Final quality validation before release.
- Subjective qualities (tone, creativity, helpfulness).
- Building ground-truth datasets.

**Protocol**: Use a structured rubric:

```python
RUBRIC = {
    "accuracy": {
        "description": "Is the factual information correct?",
        "scale": "1-5",
        "1": "Contains major factual errors",
        "3": "Minor inaccuracies present",
        "5": "Completely accurate"
    },
    "completeness": {
        "description": "Does the answer address all parts of the question?",
        "scale": "1-5",
        "1": "Ignores the question entirely",
        "3": "Addresses main point but misses details",
        "5": "Thoroughly addresses all aspects"
    },
    "clarity": {
        "description": "Is the response easy to understand?",
        "scale": "1-5",
        "1": "Confusing or incoherent",
        "3": "Adequately clear",
        "5": "Exceptionally clear and well-structured"
    },
    "safety": {
        "description": "Does the response avoid harmful content?",
        "scale": "1-5",
        "1": "Contains harmful or biased content",
        "3": "Neutral, no issues",
        "5": "Actively promotes safety and inclusion"
    },
    "adherence": {
        "description": "Does the output follow the specified format?",
        "scale": "1-5",
        "1": "Completely ignores format instructions",
        "3": "Mostly follows format with minor deviations",
        "5": "Perfectly follows all format constraints"
    }
}
```

### Automated Evaluation

Use LLMs to evaluate LLM outputs (LLM-as-a-judge). This is scalable, fast, and surprisingly correlated with human judgments.

```python
class LLMEvaluator:
    """Use a language model to evaluate prompt outputs."""
    
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        self.eval_prompt_template = """You are an expert evaluator. Assess the following response based on these criteria:

Task: {task_description}

Criteria:
{criteria}

Input: {input}

Response to evaluate: {response}

Provide a JSON evaluation:
{{
    "scores": {{"criterion_name": score_int_1_to_5}},
    "overall_score": float_average,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "suggestions": ["improvement suggestions"],
    "passed": true_or_false_if_meets_minimum_bar
}}"""
    
    def evaluate(self, input_text: str, response: str, 
                 task_description: str, criteria: dict) -> dict:
        criteria_str = "\n".join([
            f"- {name}: {info['description']} (1-5 scale)"
            for name, info in criteria.items()
        ])
        
        eval_prompt = self.eval_prompt_template.format(
            task_description=task_description,
            criteria=criteria_str,
            input=input_text,
            response=response
        )
        
        result = openai.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        return json.loads(result.choices[0].message.content)

# Usage
evaluator = LLMEvaluator()
result = evaluator.evaluate(
    input_text="Explain quantum entanglement.",
    response="Quantum entanglement is when particles are connected...",
    task_description="Explain complex scientific concepts in simple terms",
    criteria=RUBRIC
)
print(f"Overall: {result['overall_score']}/5, Passed: {result['passed']}")
```

### Automated Evaluation: Metrics

For tasks with ground truth, compute standard metrics:

```python
def compute_metrics(expected: str, actual: str) -> dict:
    """Compute similarity metrics between expected and actual outputs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Exact match
    exact_match = expected.strip().lower() == actual.strip().lower()
    
    # Token overlap (F1)
    expected_tokens = set(expected.lower().split())
    actual_tokens = set(actual.lower().split())
    intersection = expected_tokens & actual_tokens
    precision = len(intersection) / len(actual_tokens) if actual_tokens else 0
    recall = len(intersection) / len(expected_tokens) if expected_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    # Semantic similarity (using embeddings)
    def get_embedding(text: str) -> list:
        resp = openai.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding
    
    emb1 = get_embedding(expected)
    emb2 = get_embedding(actual)
    semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
    
    # ROUGE-L (simplified)
    def lcs(x: str, y: str) -> int:
        # Longest common subsequence
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_len = lcs(expected, actual)
    rouge_precision = lcs_len / len(actual) if actual else 0
    rouge_recall = lcs_len / len(expected) if expected else 0
    rouge_f1 = 2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall) if (rouge_precision + rouge_recall) else 0
    
    # Length ratio (to catch truncation or verbosity)
    length_ratio = len(actual) / len(expected) if expected else 0
    length_ok = 0.5 <= length_ratio <= 2.0
    
    return {
        "exact_match": exact_match,
        "token_f1": round(f1, 3),
        "semantic_similarity": round(semantic_sim, 3),
        "rouge_l_f1": round(rouge_f1, 3),
        "length_ratio": round(length_ratio, 2),
        "length_acceptable": length_ok,
        "overall_pass": exact_match or (f1 > 0.7 and semantic_sim > 0.85)
    }
```

---

## 3. Building an Evaluation Dataset

A good evaluation dataset covers the full spectrum of inputs your prompt will encounter.

```python
class EvalDataset:
    """Build and manage a curated evaluation dataset."""
    
    def __init__(self):
        self.test_cases = []
    
    def add_case(self, input_text: str, expected_output: str = None,
                 category: str = "general", difficulty: str = "medium",
                 metadata: dict = None):
        self.test_cases.append({
            "input": input_text,
            "expected": expected_output,
            "category": category,
            "difficulty": difficulty,
            "metadata": metadata or {}
        })
    
    def generate_synthetic(self, prompt_description: str, 
                           num_cases: int = 50, seed: str = "eval_v1") -> list[dict]:
        """Use an LLM to generate diverse synthetic test cases."""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Generate {num_cases} diverse test cases for a prompt that does:
{prompt_description}

For each case, provide:
- input: the user query/text
- expected: the correct output/answer  
- category: one of [edge_case, typical, complex, adversarial, format_variation]
- difficulty: easy/medium/hard

Make cases diverse in length, complexity, domain, and phrasing.
Include edge cases (empty input, very long input, ambiguous queries).

Return as a JSON array of {{"input": str, "expected": str, "category": str, "difficulty": str}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        
        cases = json.loads(response.choices[0].message.content)
        for c in cases:
            self.add_case(c["input"], c["expected"], 
                         c.get("category", "general"), 
                         c.get("difficulty", "medium"))
        return cases
    
    def split(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple:
        """Split into train/val/test sets stratified by category."""
        from sklearn.model_selection import train_test_split
        
        categories = [c["category"] for c in self.test_cases]
        train, temp = train_test_split(
            self.test_cases, test_size=(1 - train_ratio), 
            stratify=categories, random_state=42
        )
        
        val_cats = [c["category"] for c in temp]
        if len(set(val_cats)) > 1:
            val, test = train_test_split(
                temp, test_size=0.5, stratify=val_cats, random_state=42
            )
        else:
            val, test = temp[:len(temp)//2], temp[len(temp)//2:]
        
        return train, val, test
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.test_cases, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        ds = cls()
        with open(path) as f:
            ds.test_cases = json.load(f)
        return ds

# Build a comprehensive dataset
ds = EvalDataset()

# Add hand-crafted edge cases
ds.add_case("", "I don't see any text to process.", category="edge_case", difficulty="easy")
ds.add_case(
    "a" * 100000,
    expected_output=None,  # Expect truncation or error
    category="edge_case", 
    difficulty="hard",
    metadata={"expected_behaviour": "truncate_or_reject"}
)

# Add synthetic cases for variety
ds.generate_synthetic(
    "Classify customer reviews as positive, negative, or neutral. "
    "Extract key themes mentioned.",
    num_cases=30
)

# Save for CI pipeline
ds.save("eval_data/sentiment_classifier_v1.json")
```

---

## 4. The Eval Pipeline

Run evaluations automatically in CI/CD.

```python
class EvalPipeline:
    """Run full evaluation suite and generate reports."""
    
    def __init__(self, prompt_fn: callable, eval_dataset: list[dict],
                 evaluator: LLMEvaluator = None):
        """
        prompt_fn: function that takes input_text and returns response
        """
        self.prompt_fn = prompt_fn
        self.dataset = eval_dataset
        self.evaluator = evaluator or LLMEvaluator()
        self.results = []
    
    def run(self, parallel: bool = True, max_workers: int = 10) -> dict:
        """Run all test cases through the prompt."""
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._evaluate_single, case): case
                    for case in self.dataset
                }
                for future in as_completed(futures):
                    self.results.append(future.result())
        else:
            for case in self.dataset:
                self.results.append(self._evaluate_single(case))
        
        return self._generate_report()
    
    def _evaluate_single(self, case: dict) -> dict:
        try:
            start = time.time()
            response = self.prompt_fn(case["input"])
            latency = (time.time() - start) * 1000
            
            # Compute metrics
            if case.get("expected"):
                metrics = compute_metrics(case["expected"], response)
            else:
                metrics = {}
            
            # LLM-as-judge evaluation
            if self.evaluator:
                judge_result = self.evaluator.evaluate(
                    input_text=case["input"],
                    response=response,
                    task_description="Evaluate response quality",
                    criteria=RUBRIC
                )
            else:
                judge_result = None
            
            return {
                "case": case,
                "response": response,
                "latency_ms": round(latency, 2),
                "metrics": metrics,
                "judge": judge_result,
                "error": None
            }
            
        except Exception as e:
            return {
                "case": case,
                "response": None,
                "latency_ms": None,
                "metrics": {},
                "judge": None,
                "error": str(e)
            }
    
    def _generate_report(self) -> dict:
        passed = sum(1 for r in self.results if not r.get("error") and 
                    (not r.get("metrics") or r["metrics"].get("overall_pass", True)))
        
        # Aggregate scores
        scores = []
        for r in self.results:
            if r.get("judge") and r["judge"].get("overall_score"):
                scores.append(r["judge"]["overall_score"])
        
        avg_score = sum(scores) / len(scores) if scores else None
        
        # Per-category breakdown
        categories = {}
        for r in self.results:
            cat = r["case"].get("category", "general")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "latencies": []}
            categories[cat]["total"] += 1
            if not r.get("error"):
                categories[cat]["passed"] += 1
            if r.get("latency_ms"):
                categories[cat]["latencies"].append(r["latency_ms"])
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "avg_quality_score": avg_score,
            "avg_latency_ms": sum(
                r["latency_ms"] for r in self.results if r.get("latency_ms")
            ) / max(len([r for r in self.results if r.get("latency_ms")]), 1),
            "categories": {
                cat: {
                    "pass_rate": info["passed"] / info["total"],
                    "avg_latency_ms": sum(info["latencies"]) / len(info["latencies"]) if info["latencies"] else None
                }
                for cat, info in categories.items()
            },
            "failures": [
                {
                    "input": r["case"]["input"][:200],
                    "error": r.get("error"),
                    "expected": r["case"].get("expected"),
                    "actual": r.get("response")[:200] if r.get("response") else None,
                    "category": r["case"].get("category")
                }
                for r in self.results if r.get("error") or (
                    r.get("metrics") and not r["metrics"].get("overall_pass", True)
                )
            ],
            "timestamp": datetime.now().isoformat()
        }

# Run in CI
pipeline = EvalPipeline(
    prompt_fn=lambda x: run_my_prompt(x),
    eval_dataset=ds.test_cases
)
report = pipeline.run(parallel=True)

print(f"Pass rate: {report['pass_rate']:.1%}")
print(f"Failed: {report['failed']} cases")

# Fail CI if below threshold
if report["pass_rate"] < 0.85:
    print("FAILED: Pass rate below 85% threshold")
    exit(1)
```

---

## 5. Iterative Prompt Improvement

Use evaluation results to systematically improve prompts.

```python
class PromptOptimizer:
    """Iteratively improve prompts based on eval feedback."""
    
    def __init__(self, base_prompt: str, eval_fn: callable,
                 improvement_model: str = "gpt-4o"):
        self.current_prompt = base_prompt
        self.eval_fn = eval_fn
        self.improvement_model = improvement_model
        self.history = []
    
    def analyze_failures(self, eval_report: dict) -> str:
        """Generate a failure analysis summary."""
        if not eval_report.get("failures"):
            return "No failures to analyze."
        
        failures_summary = "\n".join([
            f"Input: {f['input'][:100]}"
            f"\nExpected: {f.get('expected', 'N/A')[:100]}"
            f"\nActual: {f.get('actual', 'N/A')[:100]}"
            f"\nError: {f.get('error', 'N/A')}"
            for f in eval_report["failures"][:10]
        ])
        
        response = openai.chat.completions.create(
            model=self.improvement_model,
            messages=[{
                "role": "user",
                "content": f"""Analyze these prompt failures and identify patterns:

Current Prompt:
{self.current_prompt}

Failures ({len(eval_report['failures'])} total):
{failures_summary}

Identify:
1. Common failure patterns
2. Root causes (ambiguous instructions, missing edge cases, format issues, etc.)
3. Specific improvements to address each pattern"""
            }],
            temperature=0
        )
        return response.choices[0].message.content
    
    def suggest_improvement(self, eval_report: dict) -> str:
        """Ask the model to suggest an improved prompt."""
        failure_analysis = self.analyze_failures(eval_report)
        
        response = openai.chat.completions.create(
            model=self.improvement_model,
            messages=[{
                "role": "user",
                "content": f"""Based on the current evaluation results and failure analysis, suggest an improved version of the prompt.

Current Prompt:
{self.current_prompt}

Evaluation Results:
- Pass rate: {eval_report.get('pass_rate', 0):.1%}
- Average quality score: {eval_report.get('avg_quality_score', 'N/A')}
- Failures by category: {json.dumps(eval_report.get('categories', {}), indent=2)}

Failure Analysis:
{failure_analysis}

Provide the improved prompt. Focus on fixing the identified failure patterns while maintaining what already works well.
Only output the new prompt, nothing else."""
            }],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    def optimize(self, max_iterations: int = 5, 
                 improvement_threshold: float = 0.02) -> list[dict]:
        """Run iterative optimization loop."""
        
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")
            
            # Evaluate current prompt
            report = self.eval_fn(self.current_prompt)
            self.history.append({
                "iteration": i + 1,
                "prompt": self.current_prompt,
                "report": report
            })
            
            print(f"Pass rate: {report['pass_rate']:.1%}")
            print(f"Quality score: {report.get('avg_quality_score', 'N/A')}")
            
            # Check if improvement is sufficient
            if i > 0:
                prev_pass = self.history[-2]["report"]["pass_rate"]
                improvement = report["pass_rate"] - prev_pass
                if improvement < improvement_threshold:
                    print(f"Improvement below threshold ({improvement:.1%}), stopping.")
                    break
            
            # Get improved prompt
            improved = self.suggest_improvement(report)
            self.current_prompt = improved
        
        return self.history
```

---

## 6. Regression Testing

Ensure prompt changes don't break existing functionality.

```python
class RegressionTestSuite:
    """Run regression tests before deploying any prompt change."""
    
    def __init__(self, baseline_responses: dict[tuple, str] = None):
        """
        baseline_responses: {(input_text, prompt_version): output}
        """
        self.baseline = baseline_responses or {}
        self.tests = []
    
    def add_regression_test(self, input_text: str, expected_output: str = None,
                            expected_behavior: str = "exact_match"):
        """
        expected_behavior: "exact_match", "contains", "semantic_similarity", 
                          "json_valid", "non_empty"
        """
        self.tests.append({
            "input": input_text,
            "expected": expected_output,
            "behavior": expected_behavior
        })
    
    def run_regressions(self, prompt_fn: callable, 
                        threshold: float = 0.85) -> dict:
        """Run regression tests and compare to baseline."""
        failures = []
        
        for test in self.tests:
            response = prompt_fn(test["input"])
            
            if test["behavior"] == "exact_match":
                passed = response.strip() == (test["expected"] or "").strip()
            elif test["behavior"] == "contains":
                passed = (test["expected"] or "") in response
            elif test["behavior"] == "semantic_similarity":
                emb1 = get_embedding(response)
                emb2 = get_embedding(test["expected"] or "")
                sim = cosine_similarity([emb1], [emb2])[0][0]
                passed = sim >= threshold
            elif test["behavior"] == "json_valid":
                try:
                    json.loads(response)
                    passed = True
                except json.JSONDecodeError:
                    passed = False
            elif test["behavior"] == "non_empty":
                passed = len(response.strip()) > 0
            else:
                passed = True
            
            if not passed:
                failures.append({
                    "input": test["input"][:100],
                    "expected": test.get("expected"),
                    "actual": response[:200],
                    "behavior": test["behavior"]
                })
        
        return {
            "total": len(self.tests),
            "passed": len(self.tests) - len(failures),
            "failed": len(failures),
            "failures": failures,
            "all_passed": len(failures) == 0
        }

# Usage in CI
def deploy_prompt():
    # Load new prompt
    new_prompt = load_prompt("prompts/classifier_v2.yaml")
    
    # Run regression tests
    suite = RegressionTestSuite()
    suite.add_regression_test("Great product!", "positive", "exact_match")
    suite.add_regression_test("Terrible service", "negative", "exact_match")
    suite.add_regression_test("It was okay.", "neutral", "exact_match")
    suite.add_regression_test("", None, "non_empty")  # Edge case
    suite.add_regression_test(
        "{"invalid json", None, "json_valid"
    )
    
    result = suite.run_regressions(new_prompt)
    
    if not result["all_passed"]:
        print(f"Regression: {result['failed']} failures!")
        for f in result["failures"]:
            print(f"  - {f['input']}: expected '{f['expected']}', got '{f['actual']}'")
        exit(1)
    
    print(f"All {result['passed']} regression tests passed!")
```

---

## 7. Continuous Monitoring in Production

Once deployed, prompts need ongoing monitoring to detect drift.

```python
class PromptMonitor:
    """Monitor prompt performance in production."""
    
    def __init__(self, prompt_name: str, 
                 quality_threshold: float = 0.8,
                 window_size: int = 1000):
        self.prompt_name = prompt_name
        self.quality_threshold = quality_threshold
        self.window_size = window_size
        self.recent_scores = deque(maxlen=window_size)
        self.daily_stats = defaultdict(list)
    
    def record_call(self, input_text: str, output: str, 
                    latency_ms: float, user_feedback: int = None):
        """
        user_feedback: optional explicit rating (1-5 thumbs up/down)
        """
        # Implicit quality signals
        output_length = len(output)
        was_truncated = output_length >= self.max_output_length if hasattr(self, 'max_output_length') else False
        
        # Compute LLM quality score asynchronously (sampled)
        quality_score = None
        if random.random() < 0.1:  # Sample 10% for cost efficiency
            quality_score = self._get_quality_score(input_text, output)
        
        record = {
            "timestamp": datetime.now(),
            "input_length": len(input_text),
            "output_length": output_length,
            "latency_ms": latency_ms,
            "user_feedback": user_feedback,
            "quality_score": quality_score,
            "was_truncated": was_truncated
        }
        
        self.recent_scores.append(record)
        
        # Daily aggregation
        day = datetime.now().strftime("%Y-%m-%d")
        self.daily_stats[day].append(record)
    
    def _get_quality_score(self, input_text: str, output: str) -> float:
        """Sample-based LLM quality evaluation."""
        result = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Rate the quality of this response (1-5):\n\nInput: {input_text}\n\nResponse: {output}\n\nQuality (1-5):"
            }],
            temperature=0,
            max_tokens=5
        )
        try:
            return float(result.choices[0].message.content.strip())
        except ValueError:
            return None
    
    def get_health_report(self) -> dict:
        """Generate a health check report."""
        if not self.recent_scores:
            return {"status": "no_data"}
        
        scores = [r for r in self.recent_scores if r["quality_score"] is not None]
        feedback = [r["user_feedback"] for r in self.recent_scores if r["user_feedback"] is not None]
        
        avg_quality = sum(s["quality_score"] for s in scores) / len(scores) if scores else None
        avg_latency = sum(r["latency_ms"] for r in self.recent_scores) / len(self.recent_scores)
        avg_feedback = sum(feedback) / len(feedback) if feedback else None
        
        # Drift detection: compare recent vs. historical average
        recent_window = list(self.recent_scores)[-100:]
        recent_quality = [
            r["quality_score"] for r in recent_window 
            if r["quality_score"] is not None
        ]
        recent_avg = sum(recent_quality) / len(recent_quality) if recent_quality else None
        
        drift_detected = False
        if avg_quality and recent_avg:
            drift_detected = recent_avg < avg_quality - 0.5
        
        return {
            "status": "degraded" if drift_detected else "healthy",
            "calls_analyzed": len(self.recent_scores),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_quality_score": round(avg_quality, 2) if avg_quality else None,
            "avg_user_feedback": round(avg_feedback, 2) if avg_feedback else None,
            "recent_quality_trend": round(recent_avg, 2) if recent_avg else None,
            "drift_detected": drift_detected,
            "p95_latency_ms": round(sorted([r["latency_ms"] for r in self.recent_scores])[
                int(len(self.recent_scores) * 0.95)
            ], 2),
            "error_rate": sum(1 for r in self.recent_scores if r.get("error")) / len(self.recent_scores)
        }
    
    def alert_if_needed(self):
        """Send alert if quality drops below threshold."""
        report = self.get_health_report()
        
        if report["status"] == "degraded":
            message = (
                f"🚨 Prompt Degradation Detected\n"
                f"Prompt: {self.prompt_name}\n"
                f"Quality dropped from {report['avg_quality_score']} to {report['recent_quality_trend']}\n"
                f"Recent latency: {report['avg_latency_ms']}ms\n"
                f"Action: Review prompt performance"
            )
            self._send_alert(message)
            return True
        return False
    
    def _send_alert(self, message: str):
        """Send to Slack/PagerDuty/etc."""
        # Implementation depends on your infrastructure
        pass

# Production usage
monitor = PromptMonitor("customer_support_classifier", quality_threshold=0.8)

# In your API handler
def handle_request(input_text: str) -> str:
    start = time.time()
    output = run_prompt(input_text)
    latency = (time.time() - start) * 1000
    
    monitor.record_call(
        input_text=input_text,
        output=output,
        latency_ms=latency,
        user_feedback=request.headers.get("X-User-Rating")
    )
    
    # Periodic health check (every 100 calls)
    if random.random() < 0.01:
        monitor.alert_if_needed()
    
    return output
```

---

## 8. Human-in-the-Loop Feedback

For high-stakes applications, incorporate human feedback directly into the improvement cycle.

```python
class HumanFeedbackCollector:
    """Collect, store, and learn from human feedback on LLM outputs."""
    
    def __init__(self, storage_path: str = "feedback/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def collect(self, prompt_input: str, model_output: str, 
                user_id: str, session_id: str) -> str:
        """Generate a feedback collection link/interface."""
        feedback_id = str(uuid.uuid4())
        
        record = {
            "feedback_id": feedback_id,
            "prompt_input": prompt_input[:1000],
            "model_output": model_output[:5000],
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": None,
            "corrected_output": None
        }
        
        with open(self.storage_path / f"{feedback_id}.json", "w") as f:
            json.dump(record, f, indent=2)
        
        return feedback_id
    
    def record_feedback(self, feedback_id: str, feedback: str,
                        corrected_output: str = None):
        path = self.storage_path / f"{feedback_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Feedback record {feedback_id} not found")
        
        with open(path) as f:
            record = json.load(f)
        
        record["feedback"] = feedback
        record["corrected_output"] = corrected_output
        record["feedback_timestamp"] = datetime.now().isoformat()
        
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
    
    def export_training_data(self, min_quality: str = "needs_improvement") -> list[dict]:
        """Export feedback as training data for prompt refinement."""
        training_pairs = []
        
        for path in self.storage_path.glob("*.json"):
            with open(path) as f:
                record = json.load(f)
            
            if record.get("corrected_output"):
                training_pairs.append({
                    "input": record["prompt_input"],
                    "original_output": record["model_output"],
                    "corrected_output": record["corrected_output"],
                    "feedback": record.get("feedback")
                })
        
        return training_pairs

feedback_collector = HumanFeedbackCollector()

# Usage
feedback_id = feedback_collector.collect(
    prompt_input="Analyze this contract clause...",
    model_output="The clause appears to be...",
    user_id="legal_team_1",
    session_id="session_abc"
)

# Later, human reviews and corrects
feedback_collector.record_feedback(
    feedback_id=feedback_id,
    feedback="Missed a key liability clause in paragraph 3",
    corrected_output="The clause is concerning because..."
)

# Export for prompt retraining
training_data = feedback_collector.export_training_data()
```

---

## 9. The Evaluation-Driven Development Workflow

```
                   ┌─────────────────────────┐
                   │  Define task & criteria  │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Build evaluation dataset│
                   │  (hand-crafted + synth)  │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Write initial prompt    │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Run evaluation pipeline │◄────┐
                   └──────────┬──────────────┘     │
                              │                    │
                   ┌──────────▼──────────────┐     │
                   │  Review results          │     │
                   │  Pass rate ≥ threshold?  │─────┤ No
                   └──────────┬──────────────┘     │
                              │ Yes                │
                   ┌──────────▼──────────────┐     │
                   │  Analyze failures        │─────┘
                   │  Identify patterns       │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Improve prompt          │
                   │  (manual or automated)   │
                   └──────────┬──────────────┘
                              │
                              │ (loop back to eval)
                              │
                   ┌──────────▼──────────────┐
                   │  Deploy to canary        │
                   │  (10% traffic)           │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Monitor production      │
                   │  quality & drift         │
                   └──────────┬──────────────┘
                              │
                   ┌──────────▼──────────────┐
                   │  Full rollout / iterate  │
                   └─────────────────────────┘
```

---

## 10. Key Metrics to Track

| Metric | What It Measures | Target | How to Measure |
|--------|-----------------|--------|----------------|
| **Pass Rate** | % of test cases that pass all checks | >85% | Eval pipeline |
| **Quality Score** | LLM-judge rating (1-5) | >4.0 | LLM-as-judge |
| **Latency (P50/P95)** | Response time | <2s / <5s | Production monitoring |
| **Cost per Call** | API cost per request | Depends on model | Cost tracker |
| **User Feedback** | Explicit user ratings | >4.0 | Feedback collection |
| **Hallucination Rate** | % factual errors in output | <5% | Human eval / LLM-as-judge |
| **Format Adherence** | % outputs matching expected format | >95% | Schema validation |
| **Drift Score** | Deviation from baseline quality | <0.5 drop | Continuous monitoring |
