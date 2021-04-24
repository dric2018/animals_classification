import os
import sys
import pandas as pd
import time

import torch as th
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import albumentations as alb
from config import Config

import vision_utils

from dataset import AnimalsDataset, DataModule
import model

import argparse

import warnings

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-type',
    '-mt',
    type=str,
    default='resnet18',
    help='Type of model architechture (present in timm package) to use, one of resnet18, efficientnet_b3 ect...'
)

parser.add_argument('--pretrained',
                    '-pt',
                    type=bool,
                    default=True,
                    help='Pretrained or not')

if __name__ == '__main__':
    # set seed for repro
    _ = seed_everything(seed=Config.seed_val)

    args = parser.parse_args()

    # get datasets
    df = pd.read_csv(os.path.join(Config.data_dir, 'dataset.csv'))
    # save experiment config
    version = vision_utils.save_experiment_conf()

    if Config.n_folds is not None:
        _ = vision_utils.run_on_folds(df=df, args=args, version=version)
    else:
        data_transforms = {
            "train": alb.Compose([
                alb.Resize(600, 600, always_apply=True),
                alb.CenterCrop(
                    height=Config.img_size,
                    width=Config.img_size,
                    always_apply=True),
                alb.HorizontalFlip(p=.6),
                alb.VerticalFlip(p=.65),
                alb.Rotate(
                    limit=35,
                    interpolation=1,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    always_apply=False,
                    p=0.43,
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.3,
                    always_apply=False,
                    p=0.5,
                ),
                alb.Normalize()

            ]),
            "test": alb.Compose([
                alb.Resize(600, 600, always_apply=True),
                alb.CenterCrop(height=Config.img_size,
                               width=Config.img_size,
                               always_apply=True),

                alb.HorizontalFlip(p=.62),
                alb.Rotate(
                    limit=35,
                    interpolation=1,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    always_apply=False,
                    p=0.33,
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.3,
                    always_apply=False,
                    p=0.58,
                ),
                alb.Normalize()

            ]),
        }
        dm = DataModule(df=df, data_transforms=data_transforms)
        print('[INFO] Setting data module up')
        dm.setup()

        # build model
        print('[INFO] Building model')

        net = model.Model(pretrained=args.pretrained)

        # config training pipeline
        print('[INFO] Callbacks and loggers configuration')
        ckpt_cb = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=Config.models_dir,
            filename=f'{Config.base_model}-{args.model_type}-version-{version}'
            + '-animals-{val_acc:.5f}-{val_loss:.5f}')

        gpu_stats = GPUStatsMonitor(memory_utilization=True,
                                    gpu_utilization=True,
                                    fan_speed=True,
                                    temperature=True)

        es = EarlyStopping(
            monitor='val_loss',
            patience=Config.early_stopping_patience,
            mode='min'
        )
        # save experiment config
        version = vision_utils.save_experiment_conf()

        Logger = TensorBoardLogger(
            save_dir=Config.logs_dir,
            name='animals',
            version=version
        )

        cbs = [es, ckpt_cb, gpu_stats]

        # build trainer
        print('[INFO] Building trainer')
        trainer = Trainer(
            gpus=1,
            precision=Config.precision,
            max_epochs=Config.num_epochs,
            callbacks=cbs,
            logger=Logger,
            deterministic=True,
            accumulate_grad_batches=Config.accumulate_grad_batches,
            fast_dev_run=False
        )

        print(f'[INFO] Runing experiment N° {version}')
        # train/eval/save model(s)
        print(f'[INFO] Training model for {Config.num_epochs} epochs')
        start = time.time()
        trainer.fit(model=net, datamodule=dm)
        end = time.time()

        duration = (end - start) / 60
        print(f'[INFO] Training time : {duration} mn')
        print("[INFO] Best loss = ", net.best_loss.cpu().item())
        print(f'[INFO] Saving model for inference')
        try:
            fn = f'animals-{Config.base_model}-version-{version}.bin'
            th.jit.save(net.to_torchscript(),
                        os.path.join(Config.models_dir, fn))
            print(f'[INFO] Model saved as {fn}')
        except Exception as e:
            print("[ERROR]", e)
