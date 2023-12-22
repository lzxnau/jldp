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

    .. card::
    """

    def __init__(
        self,
        bsize: int = 32,
        gap: int = 50,
        **kwargs: float,
    ) -> None:
        """Construct a class instance."""
        w1 = _ if (_ := kwargs.get("w1")) else 2
        w2 = _ if (_ := kwargs.get("w2")) else -3.4
        b = _ if (_ := kwargs.get("b")) else 4.2
        n = _ if (_ := kwargs.get("n")) else 0.01

        self.w = torch.tensor([w1, w2])  # w is one dimention vector
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

    1. Generate a demo training dataset.
    2. Build a linear regression model.
    3. Train the LRModel.
    4. Show and save the result.

    .. card::
    """

    def __init__(
        self,
        *,
        bsize: int = 32,
        nepoch: int = 2,
        gap: int = 50,
        **kwargs: float,
    ) -> None:
        """Construct a class instance."""
        w1 = _ if (_ := kwargs.get("w1")) else 2
        w2 = _ if (_ := kwargs.get("w2")) else -3.4
        b = _ if (_ := kwargs.get("b")) else 4.2
        n = _ if (_ := kwargs.get("n")) else 0.01
        wd = _ if (_ := kwargs.get("wd")) else 0

        data = LRData(bsize, gap, w1=w1, w2=w2, b=b, n=n)
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
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        for idx, value in enumerate(data):
            if idx > loop:
                break
            print(value)

    def plot(self) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Four Plots from DataFrame")

        axs[0, 0].plot(self.df["Weight1"])
        axs[0, 0].set_title("Weight1")

        axs[0, 1].plot(self.df["Weight2"])
        axs[0, 1].set_title("Weight2")

        axs[1, 0].plot(self.df["Bias"])
        axs[1, 0].set_title("Bias")

        axs[1, 1].plot(self.df["LossT"], label="Training Loss")
        axs[1, 1].plot(self.df["LossV"], label="Validation Loss")
        axs[1, 1].set_title("Loss")
        axs[1, 1].legend()

        plt.show()

    def fit(self) -> None:
        """
        Run a method.

        :return: None
        :rtype: None
        """
        it = 1 / self.gap
        rows = np.arange(it, self.num_epoch + it, it)
        cols = ["Weight1", "Weight2", "Bias", "LossT", "LossV"]
        self.df = pd.DataFrame(index=rows, columns=cols)
        wl1, wl2, bl, ltl, lvl = ([] for _ in range(5))
        for e in range(self.num_epoch):
            self.model.train()
            for batch in self.tloader:
                y_hat = self.model(batch[0])
                loss = mse_loss(y_hat, batch[1])
                ltl.append(loss.item())
                self.optim.zero_grad()
                with torch.no_grad():
                    loss.backward()  # type: ignore
                    self.optim.step()
                wl1.append(self.model.net.weight.tolist()[0][0])
                wl2.append(self.model.net.weight.tolist()[0][1])
                bl.append(self.model.net.bias.item())
            self.model.eval()
            for batch in self.vloader:
                with torch.no_grad():
                    y_hat = self.model(batch[0])
                    loss = mse_loss(y_hat, batch[1])
                    lvl.append(loss.item())
        self.df["LossT"] = ltl
        self.df["LossV"] = lvl
        self.df["Weight1"] = wl1
        self.df["Weight2"] = wl2
        self.df["Bias"] = bl


if __name__ == "__main__":
    bi = BaseImpl(bsize=8, nepoch=3, gap=10, w1=2, w2=-3.4, b=4.2, n=0.01)
    bi.fit()
