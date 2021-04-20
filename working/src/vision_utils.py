import os
import numpy as np
import pandas as pd
import config

import logging
logging.basicConfig(level=logging.INFO)


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
