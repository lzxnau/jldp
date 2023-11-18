# Activation Functions

Activation Functions
: [Activation functions 激活函数]{.wpg} decide whether a neuron should
  be activated or not by calculating the weighted sum and further adding
  bias to it. They are [differential 微分]{.wpg} operators for
  transforming input signals to outputs, while most of them add
  non-linearity.

ReLU Function
: ReLU provides a very simple nonlinear transformation. Given an element,
  the function is defined as the maximum of that element and 0.
  The ReLU function retains only positive elements and discards all
  negative elements by setting the corresponding activations to 0.
  When the input is negative, the [derivative 导数]{.wpg} of the ReLU
  function is 0, and when the input is positive, the derivative of the
  ReLU function is 1. It mitigated the well-documented problem of
  vanishing [gradients 梯度]{.wpg}.

  :::{math}
  \operatorname{ReLU}(x) = \max(x, 0)
  :::

Sigmoid Function
: The sigmoid function transforms those inputs whose values lie in the
  domain {math}`\mathbb{R}`, to outputs that lie on the interval (0, 1).
  The sigmoid has largely been replaced by the simpler and more easily
  trainable ReLU for most use in hidden layers. However, Sigmoids are still
  widely used as activation functions on the output units when we want to
  interpret the outputs as probabilities for binary classification problems:
  you can think of the sigmoid as a special case of the softmax.

  :::{math}
  \operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}
  :::

Tanh Function
: Like the sigmoid function, the tanh (hyperbolic tangent) function also
  squashes its inputs, transforming them into elements on the interval
  between -1 and 1. Although the shape of the function is similar to that
  of the sigmoid function, the tanh function exhibits point symmetry about
  the origin of the coordinate system.

  :::{math}
  \operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
  :::
