import os
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import torch as th

import logging
logging.basicConfig(level=logging.INFO)


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
