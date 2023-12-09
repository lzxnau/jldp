"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
import math
import random
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler


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

        self.x: Tensor = torch.randn(self.n, len(self.w))
        self.noise = torch.randn(self.n, 1) * self.noise
        wx = torch.matmul(self.x, self.w.reshape((-1, 1)))
        # reshape w to match the output of y
        self.y: Tensor = wx + self.b + self.noise  # type: ignore


class LRDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Training dataset or validation dataset for the linear regression.

    .. card::
    """

    def __init__(self, data: LRData, *, isval: bool = False) -> None:
        """Construct a class instance."""
        self.data = data
        self.isval = isval

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Subscription method for the dataset.

        :param idx: Index of the subscription.
        :type idx: int
        :retrn: Return one element from the dataset.
        :rtype: tuple[Tensor, Tensor]
        """
        if self.isval:
            idx += self.data.num_train
        return self.data.x[idx], self.data.y[idx]

    def __len__(self) -> int:
        """
        Length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        rt = self.data.num_train
        if self.isval:
            rt = self.data.num_val
        return rt


class LRSampler(Sampler[list[Tensor]]):
    """
    LRSampler Description.

    LRSampler Details.

    .. card::
    """

    def __init__(
        self,
        dataset: LRDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        """Construct a class instance."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        size = math.ceil(len(self.dataset) / self.batch_size)
        rest = len(self.dataset) / self.batch_size
        if rest > 0 and self.drop_last:
            size -= 1
        self.size = size
        self.rest = rest

    def __len__(self) -> int:
        """
        Get the size of the batches.

        :return: the length of the batches.
        :rtype: int
        """
        return self.size

    def __iter__(self) -> Iterator[list[Tensor]]:
        """
        Run a method.

        :return: None
        :rtype: None
        """
        idx = list(range(0, len(self.dataset)))
        if self.shuffle:
            random.shuffle(idx)

        for i in range(0, self.size, self.batch_size):
            try:
                ilist = idx[i : i + self.batch_size]
            except IndexError:
                ilist = idx[i : len(self.dataset)]
            bidx = torch.tensor(ilist)
            if self.dataset.isval:
                bidx += self.dataset.data.num_train  # type: ignore
            yield [self.dataset.data.x[bidx], self.dataset.data.y[bidx]]


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
        data = LRData()
        self.tdata = LRDataset(data)
        self.vdata = LRDataset(data, isval=True)
        self.tsamp = LRSampler(self.tdata, batch_size=2)
        self.loader = DataLoader(self.tdata, batch_sampler=self.tsamp)

        samp = 0
        for idx, mbatch in enumerate(self.loader):
            print(mbatch)
            if idx > samp:
                break


if __name__ == "__main__":
    bi = BaseImpl()
