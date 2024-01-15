# Classification Problems

## The Softmax

1. Purpose: Transforms a vector of real numbers into a probability
            distribution, where each value represents the probability of
            a specific class or outcome.
2. Common Usage: Final activation function in neural networks for
                 multi-class classification tasks.
3. Process:
   * Exponentiation: Applies the exponential function (e^x) to each element
                     of the input vector, emphasizing larger values.
   * Normalization: Divides each exponentiated value by the sum of all
                    exponentiated values, ensuring the output values add
                    up to 1, forming a valid probability distribution.
