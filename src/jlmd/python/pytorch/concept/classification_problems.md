# Classification Problems

## The Softmax

1. Purpose: Transforms a vector of real numbers into a probability
            distribution, where each value represents the probability of
            a specific class or outcome.
2. Usage:
   * Final activation function in neural networks for multi-class
     classification tasks.
   * Used with cross-entropy loss, a common choice for multi-class
     classification, which optimization functions aim to minimize.
3. Process:
   * Exponentiation: Applies the exponential function (e^x) to each element
                     of the input vector, emphasizing larger values.
   * Normalization: Divides each exponentiated value by the sum of all
                    exponentiated values, ensuring the output values add
                    up to 1, forming a valid probability distribution.
4. Roles:
   * Nonnegative.
   * Amplification: Exaggerates differences between input values, making
                    larger values significantly more prominent.
   * Non-linearity: Introduces non-linearity into the neural network.
   * Optimization: Differentiable nature of softmax function allows for
                   efficient use of gradient-based optimization algorithms
                   during model training.

## Cross-Entropy Loss

Entropy
: :::{math}
  H[P] = - \sum_j P(j) \log P(j)
  :::
  1. P(j): Probability of j.
  2. -log P(j): Least coding bits of P(j).

Cross-Entropy
: :::{math}
  H(P, Q) = - \sum_j P(j) \log Q(j)
  :::
  1. P(j): Real probability of j.
  2. Q(j): Predicted probability of j.
  3. For P(j), the least coding bits will be -log P(j).
  4. -log Q(j) >= -log P(j)
  5. H(P, Q) >= H(P) >= 0

Cross-Entropy Loss
: :::{math}
  l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
  :::
  1. {math}`\mathbf{y}`: Real probability distribution.
  2. {math}`\hat{\mathbf{y}}`: Predicted probability distribution.
  3. {math}`l(\mathbf{y}, \hat{\mathbf{y}})` >= {math}`l(\mathbf{y})` >= 0

Softmax
: :::{math}
  \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
  :::

Softmax and Cross-Entropy Loss
: :::{math}
  \begin{aligned}
   l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log
   \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
   &= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
   &= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
   \end{aligned}
  :::

Derivative with respect to any logit {math}`o_j`
: :::{math}
  \partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) =
  \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j =
  \mathrm{softmax}(\mathbf{o})_j - y_j
  :::

1. Mathematical Definition: H(p, y) = - Σ y_i * log(p_i)
2. y is a one-hot vector of length i: l(p, y) = -log(p_i)
3. l >= 0, when p_i = 1 = y, l = 0.
