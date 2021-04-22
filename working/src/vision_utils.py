import os
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import torch as th
import dataset

import logging
logging.basicConfig(level=logging.INFO)

# learning rate schedule params
LR_START = config.Config.lr
LR_MAX = config.Config.lr / .1
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_STEP_DECAY = 0.7
# CUSTOM LEARNING SCHEUDLE
# """
# from https://www.kaggle.com/cdeotte/how-to-compete-with-gpus-workshop#STEP-4:-Training-Schedule
# """


def ramp_scheduler(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * \
            LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)
    return lr


def view_sample(dataset: dataset.AnimalsDataset = None,
                images: th.Tensor = None,
                labels: th.Tensor = None,
                predictions: list = None,
                size=5,
                denorm=False,
                return_image=False,
                show=True):

    assert (
        labels.size(0) == len(predictions)
    ), f"Targets and predictions should have the same lengths. Label length is {labels.size(0)} while predictions is {len(predictions)}"

    tok = tokenizer.Tokenizer()

    fig = plt.figure(figsize=(size * size, size * 3))
    images, labels = images.cpu().detach(), labels.cpu().detach()
    if images is not None:
        for idx, data in enumerate(zip(images, labels, predictions)):
            img, target, prediction = data[0], data[1], data[2]

            if denorm:
                img = denormalize(img)
            else:
                img = img.transpose(1, 0).transpose(2, 1)
                img = np.clip(img, 0, 1)

            label = tok.decode(ids=target)

            ax = plt.subplot(size, size, idx + 1)

            plt.imshow(img)
            plt.title('label : ' + label, size=15)
            # plt.axis("off")
            plt.xlabel("prediction : " + prediction, size=15)
            if idx == size**2 - 1:
                if show:
                    plt.show()

                if return_image:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', orientation='portrait')
                    plt.close(fig)
                    buf.seek(0)
                    image = Image.open(buf)

                    return image
                break

    else:
        for idx, data in enumerate(dataset):
            img, target = data['img'], data['label']
            if denorm:
                img = denormalize(img)
            else:
                img = img.transpose(1, 0).transpose(2, 1)
                img = np.clip(img, 0, 1)

            label = tok.decode(ids=target)

            ax = plt.subplot(size, size, idx + 1)

            plt.imshow(img)
            plt.title('label : ' + label, size=15)
            # plt.axis("off")
            plt.xlabel("prediction : " + prediction, size=15)

            if idx == size**2 - 1:
                if show:
                    plt.show()

                if return_image:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', orientation='portrait')
                    plt.close(fig)
                    buf.seek(0)
                    image = Image.open(buf)

                    return image

                break


def denormalize(img: th.tensor, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    img = img.numpy().transpose((1, 2, 0))
    if (mean is not None) and (std is not None):
        mean = np.array(mean)
        std = np.array(std)
        img = std * img + mean
        img = np.clip(img, 0, 1)
    else:
        img = np.clip(img, 0, 1)

    return img


def get_target_from_label(label):
    for (k, v) in config.Config.classes_map.items():
        if k == label:
            return v


def get_label_from_target(target):
    for (k, v) in config.Config.classes_map.items():
        if v == target:
            return k


def show_samples(dataset: th.utils.data.Dataset,
                 size=5,
                 mean: list = [0.485, 0.456, 0.406],
                 std: list = [0.229, 0.224, 0.225]
                 ):

    plt.figure(figsize=(size*size, size*3))
    for idx, data in enumerate(dataset):
        img, target = data['x'], data['y']
        img = denormalize(img, mean=mean, std=std)

        label = get_label_from_target(target.item())
        ax = plt.subplot(size, size, idx+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

        if idx == size**2 - 1:
            break


def create_dataset_dataframe():
    dataset_size = 0
    paths = []
    labels = []

    for folder in os.listdir(config.Config.data_dir):
        logging.info(msg=f'Loading images from {folder} folder')
        images_path = os.path.join(config.Config.data_dir, folder)
        images = [
            f'{config.Config.data_dir}/{folder}/{img}' for img in os.listdir(images_path)
        ]
        logging.info(msg=f'Found {len(images)} images of {folder}')
        # print(images[:5])
        dataset_size += len(images)

        paths += images
        labels += [folder]*len(images)

    logging.info(
        msg=f'Found {dataset_size//1000}k images belonging to 10 classes'
    )
    df = pd.DataFrame({
        "path": paths,
        "label": labels
    })

    df.to_csv(
        path_or_buf=os.path.join(config.Config.data_dir, 'dataset.csv'),
        index=False
    )


if __name__ == '__main__':
    create_dataset_dataframe()
