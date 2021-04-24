import os
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import torch as th
import dataset
from tqdm.auto import tqdm
import re
import logging
import io
from PIL import Image

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


def atoi(text):
    # from  https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539
    return int(text) if text.isdigit() else text


def natural_keys(text):
    # from  https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def save_experiment_conf():

    walk = [
        folder
        for folder in os.listdir(os.path.join(config.Config.logs_dir, 'animals'))
        if ("version" in folder) and (len(folder.split('.')) <= 1)
    ]

    # sort the versions list
    walk.sort(key=natural_keys)

    if len(walk) > 0:
        version = int(walk[-1].split('version_')[-1]) + 1
    else:
        version = 0

    # save experiment config

    with open(
            os.path.join(config.Config.logs_dir, 'animals',
                         f'conf-exp-{version}.txt'), 'w') as conf:
        conf.write(
            f'================== Config file version {version} ===================\n\n'
        )
        d = dict(config.Config.__dict__)
        conf_dict = {k: d[k] for k in d.keys() if '__' not in k}

        for k in conf_dict:
            v = conf_dict[k]
            conf.write(f'{k} : {v}\n')

    return version


def make_folds(data: pd.DataFrame,
               n_folds: int = 5,
               target_col='label',
               stratified: bool = True):
    data['fold'] = 0

    if stratified:
        fold = StratifiedKFold(n_splits=n_folds,
                               random_state=config.Config.seed_value,
                               shuffle=True)
    else:

        fold = KFold(n_splits=n_folds,
                     random_state=config.Config.seed_val,
                     shuffle=True)

    for i, (tr, vr) in tqdm(enumerate(fold.split(data,
                                                 data[target_col].values)),
                            desc='Splitting',
                            total=n_folds):
        data.loc[vr, 'fold'] = i

    return data, n_folds


def view_sample(
        images: th.Tensor = None,
        labels: th.Tensor = None,
        predictions: list = None,
        size=5,
        return_image=False,
        show=True):

    image = None
    try:
        images, labels, predictions = images.cpu().detach(
        ), labels.cpu().detach(), predictions.cpu().detach()

        f, axes = plt.subplots(size, size, figsize=(size * size, size * 3))
        im_idx = 0

        for l in range(size):
            for c in range(size):
                if im_idx < images.size(0):
                    f.tight_layout(pad=1.0)
                    img = denormalize(images[im_idx])
                    label = get_label_from_target(
                        labels[im_idx].item())
                    pred = get_label_from_target(
                        predictions[im_idx].item())
                    axes[l][c].imshow(img)
                    axes[l][c].set_title(f'label : {label}')
                    axes[l][c].set_xlabel(f'prediction : {pred}')
                    axes[l][c].set_xticks([])
                    axes[l][c].set_yticks([])
                    im_idx += 1
                else:
                    axes[l][c].axis('off')
                    pass
        if show:
            plt.show()

        if return_image:
            try:
                buf = io.BytesIO()
                f.savefig(buf, format='png', orientation='portrait')
                plt.close(f)
                buf.seek(0)
                image = Image.open(buf)

            except Exception as e:
                logging.error(
                    msg=f'Could not create images grid. Reason : {e}')

    except Exception as e:
        logging.error(msg=f'{e}')

    return image


def denormalize(img: th.tensor,
                mean: list = [0.485, 0.456, 0.406],
                std: list = [0.229, 0.224, 0.225]
                ):
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
