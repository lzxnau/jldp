# Prediction Problems

:::{card}
Prediction problems can be categorized into two main types:
^^^
1. [Regression]{.wpg}: Predicting continuous numerical values, such as sales
   figures, stock prices, or weather forecasts. It refers to how much/how many
   questions.
2. [Classification]{.wpg}: Predicting discrete categorical labels, such as
   whether an email is spam or not, whether a customer will churn or not, or
   whether a patient has a specific disease or not. It refers to which on
   questions.
:::

:::{card}
In the terminology of machine learning:
^^^
1. The [dataset]{.wpg} is called a training dataset or training set.
2. Each row in the dataset is called a [sample]{.wpg}, example, data point or
   instance.
3. The thing we are trying to predict is called a [label]{.wpg} or target (Y).
4. The variables upon which the predictions are based are called
   [feature]{.wpg}s or covariates (X).
5. The [model]{.wpg} that describes how features can be transformed into an
   estimate of the target.
6. The [weight]{.wpg}s or slopes determine the influence of each feature on our
   prediction.
7. The [bias]{.wpg}, offset or intercept determines the value of the estimate
   when all features are zero.
8. [Transformation]{.wpg} involves changing the scale or form of the variables
   in order to improve the fit of the model or to make the results easier to
   interpret. For example, you might transform a variable by taking its natural
   logarithm, its square root, or its reciprocal.
9. [Translation]{.wpg} involves adding or subtracting a constant value to all
   of the observations of a variable. Translation does not change the
   relationship between the variables, but it can be used to shift the results
   of the regression model up or down. For example, you might translate a
   variable by adding or subtracting the mean of the variable.
:::

:::{card}
Linear Regression Model: {math}`{\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b`
^^^
1. {math}`\mathbf{X}`: Design Matrix
   {math}`\mathbf{X} \in \mathbb{R}^{n \times d}`. Here, {math}`\mathbf{X}`
   contains one row for every example and one column for every feature.
2. Element-wise Operations: Matrix broadcasting with vector.
3. The goal of linear regression is to find the weight vector
   {math}`\mathbf{w}` and the bias term {math}`b`.
:::

## Regression

### Linear Regression

### Softmax Regression

### Logistic Regression
