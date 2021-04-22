import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, recall, precision
from config import Config
import torchvision
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
import os
import vision_utils
import dataset
import albumentations as alb
import logging

logging.basicConfig(level=logging.INFO)


class Model(pl.LightningModule):
    def __init__(self,
                 dropout=Config.dropout_rate,
                 pretrained=False,
                 model_name=Config.base_model):
        super(Model, self).__init__()

        self.pretrained = pretrained

        self.best_f1 = -np.inf

        logging.info(
            msg=f'Using {Config.base_model} as features extractor')

        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=768,
        )

        self.decoder = nn.Linear(
            in_features=768,
            out_features=10
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad == True]

        optims = {
            "adam":
            th.optim.Adam(lr=Config.lr,
                          params=params,
                          eps=Config.eps,
                          weight_decay=Config.weight_decay),
            "adamw":
            th.optim.AdamW(lr=Config.lr,
                           params=params,
                           eps=Config.eps,
                           weight_decay=Config.weight_decay),
            "sgd":
            th.optim.SGD(lr=Config.lr,
                         params=params,
                         weight_decay=Config.weight_decay),
        }

        opt = optims[Config.optimizer]

        sc1 = th.optim.lr_scheduler.LambdaLR(optimizer=opt,
                                             lr_lambda=vision_utils.ramp_scheduler,
                                             verbose=True)

        sc2 = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='min',
            factor=0.1,
            patience=Config.reducing_lr_patience,
            threshold=0.001,
            threshold_mode='rel',
            cooldown=Config.cooldown,
            min_lr=0,
            eps=Config.eps,
            verbose=True,
        )

        if Config.reduce_lr_on_plateau:
            scheduler = sc2

            return {
                "optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        else:
            scheduler = sc1

            return [opt], [scheduler]

    def forward(self, inputs, targets=None):
        print(f'[Model info] inputs {inputs.size()}')

        features = self.encoder(inputs)
        print(f'[Model info] features {features.size()}')

        logits = self.decoder(features)
        print(f'[Model info] logits {logits.size()}')

        if targets is not None:
            print('[Model info] targets ', targets.size())

            loss = self.get_loss(logits=logits,
                                 targets=targets)

        return logits, loss

    def training_step(self, batch, batch_idx):
        images = batch['x']
        targets = batch['y']
        # forward pass + compute metrics
        log_probs, loss = self(
            inputs=images,
            targets=targets
        )

        # compute metrics
        train_acc = self.get_accuracy(pred_ids=pred_ids, targets=targets)
        train_f1 = self.get_f1(preds=preds, targets=targets)
        train_recall = self.get_recall(preds=preds, targets=targets)
        train_precision = self.get_precision(preds=preds, targets=targets)

        # logging phase

        self.log("train_acc",
                 value=train_acc,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)

        return {
            "loss": loss,
            "accuracy": train_acc,
            "f1-score": train_f1,
            "precision": train_precision,
            "recall": train_recall
        }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()
        # recall
        avg_recall = th.stack([x['recall'] for x in outputs]).mean()
        # f1
        avg_f1 = th.stack([x['f1-score'] for x in outputs]).mean()
        # precision
        avg_precision = th.stack([x['precision'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train", avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train", avg_acc,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("F1-score/Train", avg_f1,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Recall/Train", avg_recall,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Precision/Train", avg_precision,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        images = batch['x']
        targets = batch['y']
        # forward pass + compute metrics
        log_probs, val_loss = self(
            inputs=images,
            targets=targets
        )

        # compute metrics
        val_acc = self.get_accuracy(preds=preds, targets=targets)
        val_f1 = self.get_f1(preds=preds, targets=targets)
        val_recall = self.get_recall(preds=preds, targets=targets)
        val_precision = self.get_precision(preds=preds, targets=targets)

        # logging phase
        self.log("val_loss",
                 value=val_loss,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        self.log("val_acc",
                 value=val_acc,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        self.log("val_precision",
                 value=val_precision,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        self.log("val_recall",
                 value=val_recall,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        self.log("val_f1",
                 value=val_f1,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        return {
            "loss": val_loss,
            "accuracy": val_acc,
            "f1-score": val_f1,
            "recall": val_recall,
            "precision": val_precision,
            "images": images,
            "targets": targets,
            "predictions": pred_texts
        }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()
        # recall
        avg_recall = th.stack([x['recall'] for x in outputs]).mean()
        # f1
        avg_f1 = th.stack([x['f1-score'] for x in outputs]).mean()
        # precision
        avg_precision = th.stack([x['precision'] for x in outputs]).mean()
        # images
        images = th.stack([x['images'] for x in outputs])
        # targets
        targets = th.stack([x['targets'] for x in outputs])
        # predictions
        predictions = [x['predictions'] for x in outputs]

        # print(images.size())
        # print(targets.size())
        # logging using tensorboard logger
        grid = vision_utils.view_sample(images=images,
                                        labels=targets,
                                        predictions=predictions,
                                        return_image=True,
                                        show=False)

        self.logger.experiment.add_image(tag='predictions_grid',
                                         img_tensor=np.array(grid),
                                         dataformats='HWC',
                                         global_step=self.global_step)

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation", avg_acc,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("F1-score/Validation", avg_f1,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Recall/Validation", avg_recall,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Precision/Validation", avg_precision,
                                          self.current_epoch)
        # monitor f1-score improvements
        if avg_f1 > self.best_f1:
            print("\n")
            print(
                f'[INFO] validation F1-score improved from {self.best_f1} to {avg_f1}'
            )
            self.best_f1 = avg_f1
            print()
        else:
            print("\n")
            print(
                f'[INFO] validation F1-score did not improve... still at {self.best_f1}'
            )
            print()

    def predict(self, dataloader, batch_size=1):
        if batch_size == 1:
            try:
                preds = self(dataloader.unsqueeze(0))
            except:
                preds = self(dataloader)
        else:
            preds = self(dataloader)

        return preds.detach().cpu().numpy().flatten()

    def get_predictions(self, log_probs: th.Tensor) -> list:

        return [vision_utils.get_label_from_target(p) for p in log_probs.numpy()]

    def get_loss(self, logits, targets):
        logits = logits.cpu()
        targets = targets.cpu()
        # print("targets", targets.size())
        # print("logits", logits.size())
        return th.nn.CrossEntropyLoss()(input=logits, target=targets)

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return accuracy(preds=preds, target=targets)

    def get_f1(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return f1(preds=preds, target=targets, num_classes=10)

    def get_precision(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return precision(preds=preds, target=targets, num_classes=10)

    def get_recall(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return recall(preds=preds, target=targets, num_classes=10)


if __name__ == '__main__':

    df = pd.read_csv(
        os.path.join(Config.data_dir, 'dataset.csv'),
        nrows=1000
    )

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
    dm = dataset.DataModule(
        df=df,
        frac=1,
        test_size=.15,
        data_transforms=data_transforms
    )
    dm.setup()
    net = Model()
    for data in dm.val_dataloader():
        xs, ys = data['x'], data['y']
        logs, loss = net(xs, ys)
        preds = logs.softmax(dim=-1)
        print('logs', logs.size())
        print('preds', preds.size())
        precision = net.get_precision(preds=preds, targets=ys)
        acc = net.get_acc(preds=preds, targets=ys)
        f1_score = net.get_f1(preds=preds, targets=ys)
        recall = net.get_recall(preds=preds, targets=ys)

        print(f'loss={loss}')
        print(f'acc={acc}')
        print(f'precision={precision}')
        print(f'recall={recall}')
        print(f'f1_score={f1_score}')
        break
