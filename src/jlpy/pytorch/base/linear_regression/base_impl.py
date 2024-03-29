"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.23.05

"""
import math
import random
from collections.abc import Iterator
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Sampler

T = TypeVar("T")


class LRData:
    """
    Synthetic data for linear regression.

    :Ins w: Tensor-Vector --> Weights.
    :Ins b: Tensor-Scalar --> Bias.
    :Ins noise: Tensor-Vector --> Noise.
    :Ins num_train: Size --> Training set.
    :Ins num_val: Size --> Validation set.
    :Ins n: Size --> Dataset.
    :Ins x: Tensor-Matrix --> Input.
    :Ins y: Tensor-Vector --> Output.

    .. card::
    """

    def __init__(
        self,
        bsize: int = 32,
        gap: int = 50,
        ws: list[float] = [2, -3.4],
        **kwargs: float,
    ) -> None:
        """
        Construct a class instance.

        :param bsize: Batch size, default is 32.
        :type bsize: int
        :param gap: The plot gap for one epoch, default is 50.
        :type gap: int
        :param ws: List of the weights, default is [2, -3.4].
        :type ws: list[float]
        :param b: Bias, default is 4.2.
        :type b: float
        :param n: Noise, default is 0.01.
        :type n: float
        """
        b = _ if (_ := kwargs.get("b")) else 4.2
        n = _ if (_ := kwargs.get("n")) else 0.01

        self.w = torch.tensor(ws)  # w is one dimention vector
        self.b = b
        self.noise = n

        self.num_train = bsize * gap
        self.num_val = bsize * gap
        self.n = self.num_train + self.num_val  # put two sets altogether

        self.x: Tensor = torch.randn(self.n, len(self.w))
        self.noise = torch.randn(self.n, 1) * self.noise
        wx = torch.matmul(self.x, self.w.reshape((-1, 1)))
        # reshape w to match the output of y
        self.y: Tensor = wx + self.b + self.noise  # type: ignore
        # self.s: Tensor = torch.cat([self.x, self.y], dim=1)


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

    :Ins net: Linear regression layer.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        super().__init__()
        self.net = nn.LazyLinear(1)

    def forward(self, x: Tensor) -> Tensor:
        """Matrix x product with weights w plus bisa b."""
        y = self.net(x)
        if isinstance(y, Tensor):
            return y
        else:
            raise TypeError("Output  must be a Tensor.")


class BaseImpl:
    """
    Deep Learning Linear Regression Base Implemetation.

    1. Generate a demo training and validation dataset.
    2. Build a linear regression model.
    3. Train the LRModel.

        * Iterate over epoch.
        * Setup training mode.
            * Iterate over mini-batch of training set.
            * Optim.zero_grad.
            * Model forward.
            * Loss function.
            * Loss backward.
            * Optim step.
        * Setup validation mode.
            * Torch.no_grad.
            * Iterate over mini-batch of validation set.
            * Model forward.
            * Loss function.

    4. Plot the result.

    .. card::
    """

    def __init__(
        self,
        *,
        bsize: int = 32,
        nepoch: int = 2,
        gap: int = 50,
        ws: list[float] = [2, -3.4],
        **kwargs: float,
    ) -> None:
        """Construct a class instance."""
        b = _ if (_ := kwargs.get("b")) else 4.2
        n = _ if (_ := kwargs.get("n")) else 0.01
        wd = _ if (_ := kwargs.get("wd")) else 0

        data = LRData(bsize, gap, ws=ws, b=b, n=n)
        self.tdata = LRDataset(data)
        self.vdata = LRDataset(data, isval=True)
        self.tloader = DataLoader(
            self.tdata,
            batch_size=bsize,
            shuffle=True,
            drop_last=True,
            collate_fn=LRDataset.custom_collate,  # type: ignore
        )
        self.vloader = DataLoader(
            self.vdata,
            batch_size=bsize,
            shuffle=False,
            drop_last=True,
            collate_fn=LRDataset.custom_collate,  # type: ignore
        )
        self.num_epoch = nepoch
        self.gap = gap
        self.lr = 0.01
        self.model = LRModel()
        self.optim = SGD(
            [self.model.net.weight, self.model.net.bias],
            lr=self.lr,
            weight_decay=wd,
        )

    def show(self, data: DataLoader[T], loop: int = 0) -> None:
        """
        Show the dataset.

        :param data: Dataset.
        :type data: DataLoader[T]
        :param loop: Loops for iterate over dataset.
        :type loop: int
        :return: None
        :rtype: None
        """
        for idx, value in enumerate(data):
            if idx > loop:
                break
            print(value)

    def plot(self) -> None:
        """
        Plot the loss.

        :return: None
        :rtype: None
        """
        plt.plot(self.df["LossT"], label="Training Loss")
        plt.plot(self.df["LossV"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

    def fit(self) -> None:
        """
        Train the model.

        :return: None
        :rtype: None
        """
        it = 1 / self.gap
        rows = np.arange(it, self.num_epoch + it, it)
        cols = ["LossT", "LossV"]
        self.df = pd.DataFrame(index=rows, columns=cols)
        ltl, lvl = ([] for _ in range(2))
        for e in range(self.num_epoch):
            self.model.train()
            for batch in self.tloader:
                self.optim.zero_grad()
                y_hat = self.model(batch[0])
                loss = mse_loss(y_hat, batch[1])
                ltl.append(loss.item())
                loss.backward()  # type: ignore
                self.optim.step()
            self.model.eval()
            with torch.no_grad():
                for batch in self.vloader:
                    y_hat = self.model(batch[0])
                    loss = mse_loss(y_hat, batch[1])
                    lvl.append(loss.item())
        self.df["LossT"] = ltl
        self.df["LossV"] = lvl


if __name__ == "__main__":
    bi = BaseImpl(bsize=8, nepoch=3, gap=10, ws=[2, -3.4], b=4.2, n=0.01, wd=0)
    bi.fit()
