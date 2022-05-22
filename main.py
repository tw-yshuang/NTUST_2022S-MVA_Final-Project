import csv
import threading
import torch
import numpy as np
from typing import List
from torch.utils.data.dataloader import DataLoader

import src.net as net
from submodules.FileTools.WordOperator import str_format
from submodules.FileTools.PickleOperator import load_pickle
from src.data_process import get_dataloader
from src.train_process import DL_Model


class TaskThread(threading.Thread):
    def __init__(
        self,
        model: DL_Model,
        mode: str = 'train',
        loader: DataLoader or None = None,
        val_loader: DataLoader or None = None,
    ):
        threading.Thread.__init__(self)
        self.model = model
        self.mode = mode
        self.loader = loader
        self.val_loader = val_loader
        self.result = None

    def run(self):
        if self.mode == 'train':
            self.result = self.model.training(self.loader, self.val_loader)
        elif self.mode == 'test':
            self.result = self.model.testing(self.loader)

    def execute(self):
        self.start()
        try:
            self.join()
        except KeyboardInterrupt:
            self.model.earlyStop = self.model.epoch + 1


def train(model: DL_Model = None, loader: List[DataLoader] = None, data_dir: str = None, n_workers: int = 1, **kwargs):
    if model is None:
        model = DL_Model(**kwargs)

    if loader is not None:
        train_loader, valid_loader = loader
    else:
        train_loader, valid_loader, input_dim = get_dataloader(batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(input_dim=input_dim, **kwargs['net_config'])

    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


# def pre_train(model_path: str, loader: List[DataLoader] = None, data_dir: str = None, n_workers: int = 1, **kwargs):
def pre_train(model_path: str, loader: List[DataLoader] = None, **kwargs):
    model = DL_Model()

    aa = model_path.split('/')[-2].split('_')[1]
    model.net = getattr(net, 'Net_Classifier' if aa == 'Net' else aa)
    # model.net = getattr(net, model_path.split('/')[-2].split('_')[1])

    if loader is not None:
        train_loader, valid_loader = loader
    else:
        train_loader, valid_loader, input_dim = get_dataloader(batch_size=model.BATCH_SIZE, **kwargs['loader_config'])

    model.net_config(input_dim=input_dim, **kwargs['net_config'])
    model.load_model(model_path)

    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


def full_train(model: DL_Model = None, loader: DataLoader = None, data_dir: str = None, n_workers: int = 1, **kwargs):
    if model is None:
        model = DL_Model()

    if loader is None:
        loader, input_dim = get_dataloader(mode='full', batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(input_dim=input_dim, **kwargs)

    model.saveDir = './out/full/'
    task = TaskThread(model=model, loader=loader)
    task.execute()


def full_pre_train(
    model_path: str, model: DL_Model = None, loader: DataLoader = None, data_dir: str = None, n_workers: int = 1, **kwargs
):
    if model is None:
        model = DL_Model(**kwargs)

    model.load_model(model_path)

    if loader is None:
        loader, input_dim = get_dataloader(mode='full', batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(input_dim=input_dim, **kwargs)

    task = TaskThread(model=model, loader=loader)
    task.execute()


def test(model_path: str, model: DL_Model = None, loader: List[DataLoader] = None, **kwargs):
    # load original extraHyperConfig file that locate under the same folder with the model
    try:
        extraHyperConfig: dict = load_pickle(f'{model_path[: model_path.rfind("/") + 1]}/extraHyperConfig.pickle')
        extraHyperConfig.update(**kwargs)
    except FileNotFoundError:
        extraHyperConfig = kwargs

    if model is None:
        model = DL_Model(**extraHyperConfig)

    if loader is None:
        loader, labels, input_dim = get_dataloader(mode='test', batch_size=model.BATCH_SIZE, **extraHyperConfig['loader_config'])
        model.net_config(input_dim=input_dim, **extraHyperConfig['net_config'])

    model.load_model(model_path)

    task = TaskThread(mode='test', model=model, loader=loader)
    task.execute()

    with open(f'{model_path[: model_path.rfind(".pickle")]}.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Class', 'label'])

        total = 0
        acc_num = 0
        for i, (p, l) in enumerate(zip(task.result.astype(np.int32), labels)):
            writer.writerow([i, p, l])
            total += 1
            acc_num += 1 if p == l else 0

        print(f"Acc: {str_format(f'{(acc_num/total*100):.3f}', fore='y')}%")

    print(str_format("done!!", fore='g'))


if __name__ == '__main__':
    extraHyperConfig = {
        'loader_config': {
            'datasetPath': ['./Data/train_dataset_128.pickle', './Data/val_dataset_128.pickle'],
            # 'datasetPath': './Data/test_dataset_128.pickle',
            'n_workers': 8,
        },
        # 'basic_config': {},
        'net_config': {
            'network': net.VGG,
            'structure': {
                'vgg_idx': 16,
            },
            'optimizer': torch.optim.Adam,
            'learning_rate': 1e-4,
            # 'lr_scheduler': None,
        },
        # 'performance_config': {},
        # 'save_config': {},
    }

    # model = DL_Model(**extraHyperConfig)

    # train(model=model, **extraHyperConfig)

    pre_train(
        model_path='out/0520-0449_VGG_CrossEntropyLoss_Adam-1.0e-04_BS-16/0520-1322_VGG_CrossEntropyLoss_AdamW-1.0e-04_BS-16/final_e010_5.595e-02.pickle',
        **extraHyperConfig,
    )

    # full_train(**extraHyperConfig)
    # full_pre_train(**extraHyperConfig)

    # test(
    #     # model_path='out/0520-0449_VGG_CrossEntropyLoss_Adam-1.0e-04_BS-16/best-loss_e002_3.938e-02.pickle',
    #     model_path='out/0520-0501_Net_Classifier_CrossEntropyLoss_Adam-1.0e-04_BS-16/best-loss_e063_3.646e-02.pickle',
    #     **extraHyperConfig,
    # )
