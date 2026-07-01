### Definition
Generative AI (GenAI) is a branch of AI that creates new content—text, code, images, audio, video, or other data—based on patterns learned from existing data.

### High level Insight

#### Data
Everything starts with data.The better and more diverse the data, the better the model can generalize.
#### Tokens
LLMs don't understand words directly—they process tokens.The model predicts one token at a time.
example : I love programming. can be visvulaized as ["I", " love", " programming", "."]
#### Embeddings
Computers don't understand text, so words are converted into vectors (numbers).Semantically similar words have vectors that are close together.
#### Neural Networks
The intelligence comes from deep neural networks.A neural network learns patterns from data by adjusting millions or billions of parameters.
#### Transformers
The biggest breakthrough in modern GenAI.The Transformer architecture (introduced in 2017) allows models to understand relationships between words regardless of their position in a sentence.
It replaced older architectures like RNNs and LSTMs for most language tasks.
#### Attention Mechanism
Transformers use self-attention.Instead of reading text strictly left to right, the model determines which words are most relevant to each other.
example : John put the book on the table because it was heavy.
The model learns what "it" refers to using attention.
#### Pre-training
Models first learn from enormous amounts of unlabeled data.They repeatedly perform a simple task:
Predict the next token.
Example:
The capital of France is ___
The model learns language, grammar, reasoning patterns, and factual associations during this stage.
#### Fine-tuning
After pre-training, a model can be specialized.
Examples:
- Medical assistant
- Legal assistant
- Coding assistant
- Customer support chatbot
  Fine-tuning adapts the model to a particular domain or task.

#### Prompting
A prompt tells the model what to do.Prompt quality often has a significant impact on output quality.
#### Context Window
A model can only consider a limited amount of information at once.That limit is called the context window.
To get the most out of your context window, it is a best practice to keep your prompts structured and put the most critical instructions at the very beginning.
#### Inference
Inference is when a trained model generates responses for new inputs.Training teaches the model; inference is using what it learned.
#### Hallucinations
Models sometimes generate confident but incorrect information.
Reducing hallucinations often involves:
- Better prompts
- Grounding with external data (RAG)
- Tool use
- Human verification


