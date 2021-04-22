import torch as th
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import config
import albumentations as alb


class AnimalsDataset(Dataset):
    def __init__(self, df,
                 data_dir: str = config.Config.data_dir,
                 task='train',
                 transform=None):

        self.df = df
        self.images_dir = data_dir
        self.task = task
        self.transform = transform

        # print(self.images_dir)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = self.df.iloc[index].path
        img = Image.open(img_path)
        img = np.array(img)
        # apply transforms if not none
        if self.transform is not None:
            img = self.transform(image=img)['image']
            img = th.from_numpy(img.transpose((2, 0, 1)))
        else:
            # transform to tensor and normalize
            img = th.from_numpy(img.transpose((2, 0, 1))).float() / 255.

        sample = {
            'x': img,  # image tensor

        }

        if self.task == 'train':
            label = self.df.iloc[index].label
            target = config.Config.classes_map[label]
            sample.update({
                'y': th.tensor(target, dtype=th.long)
            })

        return sample


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 df: pd.DataFrame,
                 data_transforms=None,
                 frac: float = 0,
                 train_batch_size: int = config.Config.train_batch_size,
                 test_batch_size: int = config.Config.train_batch_size,
                 test_size: float = .1,
                 n_classes: int = 10
                 ):
        super(DataModule, self).__init__()
        self.frac = frac
        self.df = df
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_size = test_size
        self.n_classes = n_classes
        self.data_transforms = data_transforms

    def setup(self, stage=None):
        # datasets
        # if fraction is fed
        train_df, val_df = train_test_split(self.df, test_size=self.test_size)

        if self.frac > 0:
            train_df = train_df.sample(frac=self.frac).reset_index(drop=True)
            self.train_ds = AnimalsDataset(df=train_df,
                                           task='train',
                                           transform=self.data_transforms['train'])
        else:
            self.train_ds = AnimalsDataset(df=train_df,
                                           task='train',
                                           transform=self.data_transforms['train'])

        self.val_ds = AnimalsDataset(df=val_df,
                                     task='train',
                                     transform=self.data_transforms['test'])

        training_data_size = len(self.train_ds)
        validation_data_size = len(self.val_ds)

        print(
            f'[INFO] Training on {training_data_size} samples belonging to {self.n_classes} classes')
        print(
            f'[INFO] Validating on {validation_data_size} samples belonging to {self.n_classes} classes')

    # data loaders
    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=config.Config.num_workers,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=config.Config.num_workers,
                          pin_memory=False)


if __name__ == '__main__':

    df = pd.read_csv(os.path.join(config.Config.data_dir, 'dataset.csv'))
    data_transforms = {
        "train": alb.Compose([
            alb.Resize(height=config.Config.img_size,
                       width=config.Config.img_size)
        ]),
        "test": alb.Compose([
            alb.Resize(height=config.Config.img_size,
                       width=config.Config.img_size)
        ]),
    }
    dm = DataModule(
        df=df,
        frac=1,
        test_size=.15,
        data_transforms=data_transforms
    )
    dm.setup()

    for data in dm.val_dataloader():
        xs, ys = data
        print(xs.size())
        print(ys.size())
