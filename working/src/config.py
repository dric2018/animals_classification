import os


class Config:
    base_dir = '/home/zeusdric/Dric/DataScience/projects/animals/'
    wording_dir = os.path.join(base_dir, 'working')
    data_dir = os.path.join(base_dir, 'data/raw-img')
    train_batch_size = 8
    test_batch_size = 8
    lr = 1e-3
    num_epochs = 30
    n_folds = 5
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    device = 'cuda'
    img_size = 300
    n_channels = 3
    base_model = 'resnet34'
    optimizer = 'adamw'
    eps = 1e-08
    weight_decay = 1e-2
    precision = 32
    num_workers = 2
    dropout_rate = .2
    classes_map = {
        "butterfly": 0,
        "cat": 1,
        "chicken": 2,
        "cow": 3,
        "dog": 4,
        "elephant": 5,
        "horse": 6,
        "sheep": 7,
        "spider": 8,
        "squirrel": 9
    }
