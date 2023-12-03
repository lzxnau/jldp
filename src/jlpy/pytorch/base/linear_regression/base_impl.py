"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
import torch
from torch import nn


class LRModel(nn.Module):
    """Custom Linear Regression Model as a torch.nn.Module."""

    def __init__(self, sigma: float = 0.01) -> None:
        """Construct a class instance."""
        super().__init__()
        self.w = torch.normal(0, sigma, (100, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        print(self.b)


if __name__ == "__main__":
    lrm = LRModel()
