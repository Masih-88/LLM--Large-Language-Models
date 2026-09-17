Attention and Transformer Study Notes

A structured technical study of the Transformer architecture, combining the mathematical foundations, architectural components, computational complexity, and PyTorch implementations behind modern Transformer-based models.

This repository documents the core concepts introduced in the paper “Attention Is All You Need” and connects the mathematical equations with practical PyTorch implementations.

📚 Contents
Overview
Transformer Architecture
Encoder–Decoder Structure
Scaled Dot-Product Attention
Multi-Head Attention
Positional Encoding
Position-wise Feed-Forward Network
Residual Connections & Layer Normalization
Transformer Data Flow
Computational Complexity
PyTorch Implementation
Key Hyperparameters
Learning Objectives
References
🔍 Overview

Transformers are neural network architectures designed to model relationships between elements in sequential data using attention mechanisms, rather than relying on recurrence.

The original Transformer architecture consists of:

Encoder stack
Decoder stack
Multi-head self-attention
Scaled dot-product attention
Position-wise feed-forward networks
Positional encodings
Residual connections
Layer normalization

The original architecture uses:

Parameter	Value
Number of encoder layers	6
Number of decoder layers	6
Model dimension (d_model)	512
Attention heads (h)	8
Key/Query dimension (d_k)	64
Value dimension (d_v)	64
Feed-forward dimension (d_ff)	2048
🏗️ Transformer Architecture

The original Transformer follows an encoder–decoder architecture.

The encoder converts the input sequence into contextual representations. The decoder then generates the output sequence while attending to both previously generated tokens and the encoder representations.

High-Level Structure
Input Tokens
     │
     ▼
Token Embeddings
     │
     + Positional Encoding
     │
     ▼
┌──────────────────────┐
│   Encoder × N        │
│                      │
│ Multi-Head Attention │
│         ↓            │
│ Add & Norm            │
│         ↓            │
│ Feed-Forward Network │
│         ↓            │
│ Add & Norm            │
└──────────────────────┘
     │
     ▼
Encoder Representations
     │
     ▼
┌──────────────────────┐
│   Decoder × N        │
│                      │
│ Masked Self-Attention│
│         ↓            │
│ Add & Norm            │
│         ↓            │
│ Cross-Attention      │
│         ↓            │
│ Add & Norm            │
│         ↓            │
│ Feed-Forward Network │
│         ↓            │
│ Add & Norm            │
└──────────────────────┘
     │
     ▼
Linear Layer
     │
     ▼
Softmax
     │
     ▼
Output Probabilities
🧱 Encoder–Decoder Structure

The original Transformer uses N = 6 identical encoder layers and N = 6 identical decoder layers.

Each encoder layer contains:

Multi-head self-attention
Add & Norm
Position-wise feed-forward network
Add & Norm

Each decoder layer contains:

Masked multi-head self-attention
Add & Norm
Encoder–decoder attention
Add & Norm
Position-wise feed-forward network
Add & Norm

Residual connections allow information to flow through the network:

[
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
]

🎯 Scaled Dot-Product Attention

Attention is the central mechanism of the Transformer.

The core equation is:

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Where:

(Q) = Query matrix
(K) = Key matrix
(V) = Value matrix
(d_k) = dimensionality of the keys
Step 1 — Calculate Similarity

[
QK^T
]

The matrix multiplication measures how strongly each query relates to each key.

Step 2 — Scale

[
\frac{QK^T}{\sqrt{d_k}}
]

The scaling factor prevents the dot products from becoming excessively large when (d_k) increases.

Step 3 — Normalize with Softmax

[
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
]

This produces attention weights.

Step 4 — Weighted Combination

\text{Attention Weights} \times V
]

The resulting representation contains information from the value vectors weighted according to their relevance.

🧠 Multi-Head Attention

Instead of performing a single attention operation, Transformers perform attention across multiple independent heads.

For the original Transformer:

[
h = 8
]

and:

[
d_k = d_v = 64
]

because:

[
8 \times 64 = 512 = d_{\text{model}}
]

Each head learns a different representation of relationships between tokens.

Multi-Head Attention

\text{Concat}(head_1,\ldots,head_h)W^O
]

where:

[
head_i =
\text{Attention}
(QW_i^Q,KW_i^K,VW_i^V)
]

This allows the model to jointly attend to information from different representation subspaces.

Conceptual Example
                  Input
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Head 1        Head 2       Head 3   ... Head 8
       │            │            │           │
       ▼            ▼            ▼           ▼
  Attention     Attention    Attention   Attention
       │            │            │           │
       └────────────┴────────────┴───────────┘
                    │
                 Concatenate
                    │
                    ▼
              Linear Projection
                    │
                    ▼
                  Output
📍 Positional Encoding

Unlike recurrent neural networks, Transformers do not inherently process tokens sequentially.

Therefore, information about token position must be injected into the input representations.

The original Transformer uses sinusoidal positional encoding.

For even dimensions:

\sin
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
]

For odd dimensions:

\cos
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
]

Where:

(pos) = token position
(i) = dimension index
(d_{\text{model}}) = embedding dimension

The positional encoding is added directly to the token embeddings:

[
X =
Embedding(tokens) + PE
]

This allows the model to incorporate information about token order without recurrence.

⚙️ Position-wise Feed-Forward Network

Each Transformer encoder and decoder layer contains a fully connected feed-forward network.

The original architecture uses:

[
d_{\text{model}} = 512
]

and expands the representation to:

[
d_{ff} = 2048
]

The transformation is:

\max(0,xW_1+b_1)W_2+b_2
]

The first linear transformation expands the representation:

[
512 \rightarrow 2048
]

The second projects it back:

[
2048 \rightarrow 512
]

The same feed-forward network is applied independently to every position.

🔄 Residual Connections & Layer Normalization

Transformer sublayers use residual connections.

Conceptually:

Input
  │
  ├───────────────┐
  │               │
  ▼               │
Sublayer          │
  │               │
  ▼               │
  + ◄─────────────┘
  │
  ▼
Layer Normalization
  │
  ▼
Output

The residual connection can be expressed as:

[
x + Sublayer(x)
]

followed by normalization.

Residual connections help information and gradients propagate through deep Transformer stacks.

🔁 Transformer Data Flow

A simplified encoder data flow is:

Tokens
   │
   ▼
Embedding
   │
   +
Positional Encoding
   │
   ▼
Multi-Head Self-Attention
   │
   ▼
Add & Norm
   │
   ▼
Feed-Forward Network
   │
   ▼
Add & Norm
   │
   ▼
Encoder Output

The decoder additionally performs cross-attention:

Decoder Input
     │
     ▼
Masked Self-Attention
     │
     ▼
Add & Norm
     │
     ▼
Cross-Attention ◄──── Encoder Output
     │
     ▼
Add & Norm
     │
     ▼
Feed-Forward
     │
     ▼
Add & Norm
     │
     ▼
Linear
     │
     ▼
Softmax
     │
     ▼
Output
⏱️ Computational Complexity

One of the major advantages of the Transformer architecture is that its attention operations can be computed in parallel across sequence positions.

For self-attention, the major matrix multiplication has approximately:

[
O(n^2d)
]

complexity, where:

(n) = sequence length
(d) = representation dimension

The quadratic dependence on sequence length is an important computational consideration for long sequences.

Comparison with Recurrent Architectures
Property	RNN	Transformer
Sequential computation	Required	Highly parallelizable
Path between distant tokens	(O(n))	(O(1))
Self-attention	No	Yes
Parallel training	Limited	High
Long-range dependencies	More difficult	Direct attention path
Attention complexity	—	(O(n^2d))

The Transformer therefore trades recurrent sequential computation for highly parallel attention operations, while introducing quadratic attention complexity with respect to sequence length.

🐍 PyTorch Implementation

The repository also contains modular PyTorch implementations of the major Transformer components.

A simplified scaled dot-product attention implementation:

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / torch.sqrt(
        torch.tensor(d_k, dtype=Q.dtype)
    )

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights

The implementation follows the mathematical definition:

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

🧩 Modular Implementation Structure

A practical PyTorch implementation can be organized into independent components:

Transformer/
│
├── embeddings.py
├── positional_encoding.py
├── attention.py
├── multi_head_attention.py
├── feed_forward.py
├── encoder.py
├── decoder.py
├── transformer.py
│
├── notebooks/
│   └── transformer_experiments.ipynb
│
├── notes/
│   └── transformer-study-notes-v3.pdf
│
└── README.md

This structure separates the mathematical concepts into reusable implementation modules.

📊 Key Hyperparameters
Component	Symbol	Original Value
Encoder layers	(N)	6
Decoder layers	(N)	6
Model dimension	(d_{model})	512
Attention heads	(h)	8
Key dimension	(d_k)	64
Value dimension	(d_v)	64
Feed-forward dimension	(d_{ff})	2048
🎓 Learning Objectives

This study is designed to connect theory → mathematics → implementation.

After completing these notes, the learner should be able to:

Explain the overall Transformer architecture.
Understand the role of encoder and decoder stacks.
Derive and interpret scaled dot-product attention.
Explain Query, Key, and Value representations.
Understand why attention is scaled by (\sqrt{d_k}).
Explain how multi-head attention works.
Implement attention mechanisms using PyTorch.
Understand sinusoidal positional encoding.
Explain position-wise feed-forward networks.
Understand residual connections and layer normalization.
Analyze Transformer computational complexity.
Connect Transformer equations to actual neural-network code.
🧪 Theory → Code Connection

One of the main goals of this project is to avoid treating Transformer equations as isolated mathematical formulas.

Each major component is connected to its corresponding implementation:

Theory	Mathematical Concept	Implementation
Attention	(QK^T)	Matrix multiplication
Scaling	(1/\sqrt{d_k})	Tensor division
Attention weights	Softmax	F.softmax()
Weighted values	(AV)	Matrix multiplication
Multi-head attention	Multiple projections	Linear layers
Positional encoding	Sine/Cosine	Tensor operations
FFN	Two linear layers	nn.Linear
Residual connection	(x + f(x))	Tensor addition
Normalization	Layer normalization	nn.LayerNorm

This theory-to-code mapping is useful for understanding not only how Transformers work, but also how they are implemented in modern deep-learning frameworks.

📖 Reference

The primary architectural reference for this study is:
https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/nbrpv/the-transformer-block
https://www.aparat.com/v/vfh69s9
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1706.03762

Vaswani et al., “Attention Is All You Need”, 2017.

The study notes combine mathematical explanations, architectural diagrams, computational analysis, and PyTorch implementations to provide a practical learning reference.

📁 Included Study Material
transformer-study-notes-v3.pdf

The PDF contains the detailed mathematical derivations, structural explanations, and implementation-focused notes supporting this repository.

🚀 Purpose of This Repository

This project is part of a broader effort to develop a strong understanding of Deep Learning, Natural Language Processing, attention and Transformer architectures by studying the underlying mathematics and implementing the concepts directly in PyTorch.

Rather than treating Transformers as a black box, the objective is to understand the architecture at three levels:

Mathematics
     ↓
Architecture
     ↓
PyTorch Implementation

Core principle: Understand the equation → understand the architecture → implement it from scratch.
