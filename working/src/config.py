import os


class Config:
    base_dir = '/home/zeusdric/Dric/DataScience/projects/animals/'
    wording_dir = os.path.join(base_dir, 'working')
    data_dir = os.path.join(base_dir, 'data/raw-img')
    seed_val = 21
    train_batch_size = 80
    test_batch_size = 32
    lr = 3e-1
    num_epochs = 3
    n_folds = None
    stratified = True
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    device = 'cuda'
    img_size = 300
    n_channels = 3
    base_model = 'resnet18'
    optimizer = 'adamw'
    reduce_lr_on_plateau = True
    reducing_lr_patience = 8
    early_stopping_patience = 10
    eps = 1e-08
    weight_decay = 1e-2
    precision = 32
    accumulate_grad_batches = 2
    cooldown = .0
    num_workers = 3
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
