"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
from torch import nn


class LRModel(nn.Module):
    """Custom Linear Regression Module."""

    def __init__(self) -> None:
        """Construct a class instance."""
        super().__init__()
        print("this is a test")


if __name__ == "__main__":
    lrm = LRModel()
