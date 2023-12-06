"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader


class LRData:
    """
    Synthetic data for linear regression.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        self.w = torch.tensor([2, -3.4])  # w is one dimention vector
        self.b = 4.2
        self.noise = 0.01

        self.num_train = 1000
        self.num_val = 1000
        self.n = self.num_train + self.num_val  # put two sets altogether

        self.x = torch.randn(self.n, len(self.w))
        self.noise = torch.randn(self.n, 1) * self.noise
        self.y = (
            torch.matmul(self.x, self.w.reshape((-1, 1))) + self.b + self.noise
            # reshape w to match the output of y
        )


class LRModel(nn.Module):
    """
    Custom Linear Regression Model as a torch.nn.Module.

    .. card::
    """

    def __init__(
        self, features: int = 10, lr: float = 0.03, sigma: float = 0.01
    ) -> None:
        """Construct a class instance."""
        super().__init__()
        self.loss = nn.MSELoss()
        self.lr = lr
        self.w: Tensor = torch.normal(
            0, sigma, (features, 1), requires_grad=True
        )
        self.b: Tensor = torch.zeros(1, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Matrix x product with weights w plus bisa b."""
        y: Tensor = torch.matmul(x, self.w) + self.b
        return y

    def backward(self) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        ...


class BaseImpl:
    """
    Deep Learning Linear Regression Base Implemetation.

    1. Generate a demo training dataset.
    2. Build a linear regression model.
    3. Train the LRModel.
    4. Show and save the result.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        ...


if __name__ == "__main__":
    lrm = LRModel()
