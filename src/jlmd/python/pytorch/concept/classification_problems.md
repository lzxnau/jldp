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
