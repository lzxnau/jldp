"""
Lions and cheetahs image classification example.

Classify lion or cheetah images using a pre-trained Deep Learning model with
the PyTorch framework.

:Authors: JLPy
:Version: 2023.11.26.03

"""
# start import
import math
import os
import random as rand
from os.path import isfile, join

import albumentations as al  # type: ignore
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
from torch import FloatTensor
from torch.utils.data import DataLoader, Dataset

# end import


class Cfg:
    """
    Static configurations for the project.

    :Cls img_path: The directory of the images under the project folder.
    :Cls sub_dirs: A list of the image directories for lions and cheetahs.
    :Cls labels: 0 refers to lions and 1 refers to cheetahs.
    :Cls img_size: The standard image size for feeding the model.
    :Cls img_df: A pandas DataFrame to store data information.
    :Cls batch_size: The size of a batch samplers.

    .. attention::
       | The Python working directory will be the same as the Jupyter working
         directory.
       | img_path will use the relative path to locate the source images.
       | After this setup, the outputs from both Python and Jupyter will work
         properly.

    .. card::
    """

    if jldp := os.environ.get("JLDPDIR"):
        img_path = jldp + "/res/images"
    else:
        img_path = "../res/images"
    sub_dirs = ("Lions", "Cheetahs")
    labels = (0, 1)
    img_size = 256
    img_df = pd.DataFrame([])
    batch_size = 8

    @classmethod
    def load_df(cls) -> None:
        """
        Load data into a pandas dataframe.

        From the local drive, extract all file paths and their corresponding
        labels, and save them in a pandas DataFrame with two columns: file_path
        and label.
        """
        data = []
        sub_paths = [join(cls.img_path, x) for x in cls.sub_dirs]
        for s, la in zip(sub_paths, cls.labels):
            for f in os.listdir(s):
                if ".jpg" in f and isfile(p := join(s, f)):
                    data.append((p, la))
        cls.img_df = pd.DataFrame(data, columns=["file_path", "label"])

    @classmethod
    def get_df(cls) -> pd.DataFrame:
        """Return class attribute."""
        if cls.img_df.empty:
            cls.load_df()
        return cls.img_df


class Explore:
    """
    Prepare and Explore Data.

    This is the first step to explore the source data and to become familiar
    with the data.

    .. card::
    """

    @staticmethod
    def cnt_imgs() -> None:
        """Count images by the category of the labels."""
        print(Cfg.get_df().groupby(["label"]).count())
        sns.countplot(Cfg.get_df(), x="label")

    @staticmethod
    def samp_imgs() -> None:
        """
        Show sample images from the category of each label.

        1. Grouping the dataframe by the column 'label'.
        2. Sampling 3 images from each group.
        3. Shuffling the order of the new dataframe.
        4. Creating a plot with 3x2 figures.
        5. Loading the sample images.
        6. Plotting the figures.
        """
        # Step 1
        dfg = Cfg.get_df().groupby("label")

        # Step 2
        dfs = dfg.apply(lambda x: x.sample(3, replace=False))
        dfs = dfs.reset_index(drop=True)

        # Step 3
        dfs = dfs.sample(frac=1).reset_index(drop=True)

        # Step 4
        _, ax = plt.subplots(2, 3, figsize=(10, 6))

        # Step 5
        for i in range(2):
            for j in range(3):
                label = dfs.label[i * 3 + j]
                file_path = dfs.file_path[i * 3 + j]

                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))

                ax[i, j].imshow(image)
                ax[i, j].set_title(
                    f"Label: {label} ({'Lion' if not label else 'Cheetah'})"
                )
                ax[i, j].axis("off")

        # Step 6
        plt.tight_layout()
        plt.show()


class ImgsDataset(Dataset[tuple[FloatTensor, list[int]]]):
    """A custom subclass of Dataset for loading the images."""

    def __init__(self) -> None:
        """ImgsDataset constractor."""
        mdf = Cfg.get_df()
        self.file_paths = mdf.loc[:, "file_path"].to_numpy()
        self.labels = mdf.loc[:, "label"].to_numpy()
        self.transform = al.Compose(
            [
                al.Resize(Cfg.img_size, Cfg.img_size),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        """
        Get the size of the dataset.

        :return: The size of the dataset.
        :rtype: int
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, list[int]]:
        """
        Get one element of the dataset.

        :param idx: Index of the element in the dataset.
        :return: One element of the dataset with a format (tensor, label).
        :rtype: tuple[tensor, int]
        """
        label = self.labels[idx]
        image = cv2.imread(self.file_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = image / 255
        return image, label

    def samp_demo(self) -> None:
        """
        Print five demo samples from the ImgsDataset.

        Optional method for testing purposes.
        """
        s = math.floor(len(self.labels) / 5)
        for i in range(5):
            j = rand.randint(i * s, i * s + s)
            print(self[j])

    def batch_demo(self) -> None:
        """
        Print demo batch info from the Dataloader.

        Optional method for testing purposes.
        """
        dl = DataLoader(
            self, batch_size=Cfg.batch_size, shuffle=True, num_workers=0
        )
        for batch_img, batch_label in dl:
            print(batch_img.shape)
            print(batch_label.shape)
            break


if __name__ == "__main__":
    Explore.cnt_imgs()
    Explore.samp_imgs()
    # ids = ImgsDataset()
    # ids.samp_demo()
