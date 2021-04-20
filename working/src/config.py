import os


class Config:
    base_dir = '/home/zeusdric/Dric/DataScience/projects/animals/'
    wording_dir = os.path.joing(base_dir, 'working')
    data_dir = os.path.joing(base_dir, 'data')
    train_batch_size = 32
    test_batch_size = 32
    lr = 1e-4
    num_epochs = 30
    n_folds = 5
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    device = 'cuda'
    img_size = 300
    n_channels = 3
    base_model = 'resnet50'
    precision = 32
