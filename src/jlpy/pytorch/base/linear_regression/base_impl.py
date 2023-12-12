"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.10.02

"""
import math
import random
from collections.abc import Iterator
from typing import TypeVar

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

T = TypeVar("T")


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
        self.s: Tensor = torch.cat([self.x, self.y], dim=1)


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

    def __getitems__(self, idx: list[int]) -> tuple[Tensor, Tensor]:
        """
        Batch subscription method for the dataset.

        :param idx: A list of indices.
        :type idx: list[int]
        :retrn: Return a batch of elements from the dataset.
        :rtype: tuple[Tensor, Tensor]
        """
        idx = torch.tensor(idx, dtype=torch.long)
        if self.isval:
            idx += self.data.num_train  # type: ignore
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

    @staticmethod
    def custom_collate(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Collate batch input to output without any changes.

        * Default collate:
            * Merge a list of one element from the dataset to a Tensor \
              collection format.

        * Custom collate:
            * If there is a way to load batched samplers from database, \
              bypass the default collate method.

        :param batch: Batch input returned from __getitems__().
        :type batch: tuple[Tensor, Tensor]
        :return: Return the original input.
        :rtype: tuple[Tensor, Tensor]
        """
        return batch


class LRSampler(Sampler[list[int]]):
    """
    A custom torch batched Sampler - LRSampler.

    * Custom Sampler:
        * Controlling how to shuffle.
        * Return the size for iteration.

    * Custom batched sampler:
        * Control how to shuffle.
        * Control batch size.
        * Control the drop_last.
        * Return the size for batched iteration.

    * Sampler __iter__():
        * Return a list of indices.
        * Using the list of indices to get elements from Dataset class.
        * Sampler will call __getitems__() first, then fall back to call \
          __getitem__().

    * Dataset class:
        * Sampler class only need to get the len of the dataset.
        * According to the len of the dataset, Sampler will procee all the \
          rest itself.
        * Dataloader will iterate the Sampler and get the samples from Dataset.

    .. card::
    """

    def __init__(
        self,
        dsize: int,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        """Construct a class instance."""
        self.dsize = dsize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        size = math.ceil(self.dsize / self.batch_size)
        rest = dsize / self.batch_size
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

    def __iter__(self) -> Iterator[list[int]]:
        """
        Iterate to get elements from Dataset.

        :return: Return a list of indices.
        :rtype: Iterator[list[int]]
        """
        idx = list(range(0, self.dsize))
        if self.shuffle:
            random.shuffle(idx)

        for i in range(0, self.size, self.batch_size):
            try:
                ilist = idx[i : i + self.batch_size]
            except IndexError:
                ilist = idx[i : self.dsize]
            yield ilist


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
        self.tloader = DataLoader(
            self.tdata,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            collate_fn=LRDataset.custom_collate,  # type: ignore
        )
        self.vloader = DataLoader(
            self.vdata,
            batch_size=32,
            shuffle=False,
            drop_last=True,
            collate_fn=LRDataset.custom_collate,  # type: ignore
        )

    def show(self, data: DataLoader[T], samp: int = 0) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        for idx, value in enumerate(data):
            if idx > samp:
                break
            print(value)

    def fit(self) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        ...

    def demo(self) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        self.show(self.tloader)
        self.show(self.vloader)


if __name__ == "__main__":
    bi = BaseImpl()
    bi.demo()
