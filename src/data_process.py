from __future__ import barry_as_FLUFL
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_preprocess import preprocess


# sys.path.append(os.path.abspath(__package__))
from submodules.FileTools.FileSearcher import get_filenames, str_format
from submodules.FileTools.PickleOperator import load_pickle

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class TrafficLightDataset(Dataset):
    def __init__(self, data, labels=None, transforms=train_transforms):
        self.data = data
        self.labels = None
        self.transforms = transforms
        if labels is not None:
            aa: torch.Tensor

            self.labels = F.one_hot(torch.tensor(labels), num_classes=3).type(torch.float16)
            # F.one_hot(torch.tensor(self.label), num_classes=3)

    def __getitem__(self, index: int):
        x = self.transforms(self.data[index])
        if self.labels is not None:
            return self.transforms(self.data[index]), self.labels[index]
        else:
            return self.transforms(self.data[index])

    def __len__(self):
        return len(self.data)


def get_dataloader(
    mode: str = 'train',
    data_dir: str = None,
    ann_path: str = None,
    datasetPath: str = None,
    imgShape: Tuple[int] = None,
    batch_size=32,
    n_workers=1,
    **kwargs
):
    assert [data_dir, ann_path, datasetPath, imgShape] != [None] * 4, print(str_format("all none!!", fore="r"))
    mode = mode.lower()
    if mode == 'test':
        if datasetPath is not None:
            data, labels = load_pickle(datasetPath)

        elif not None in [data_dir, ann_path, imgShape]:
            data, labels = preprocess(data_dir, ann_path, begin_rate=0.9, end_rate=1.0, imgShape=imgShape)

        return (
            DataLoader(
                dataset=TrafficLightDataset(data),
                batch_size=batch_size,
                num_workers=n_workers,
            ),
            labels,
            3,
        )

    elif mode == 'full':
        if datasetPath is not None:
            data, labels = load_pickle(datasetPath)
        elif None in [data_dir, ann_path, imgShape]:
            data, labels = preprocess(data_dir, ann_path, begin_rate=0.0, end_rate=1.0, imgShape=imgShape)

        # labels = F.one_hot(torch.tensor(labels), num_classes=3)
        return (
            DataLoader(
                dataset=TrafficLightDataset(data, labels),
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                num_workers=n_workers,
            ),
            3,
        )

    else:
        if type(datasetPath) is list:
            train_loader, input_dim = get_dataloader(
                mode='full', datasetPath=datasetPath[0], batch_size=batch_size, n_workers=n_workers
            )
            val_loader, input_dim = get_dataloader(mode='full', datasetPath=datasetPath[1], batch_size=batch_size, n_workers=n_workers)
        else:
            train_data, train_labels = preprocess(data_dir, ann_path, begin_rate=0.0, end_rate=0.7, imgShape=imgShape)
            val_data, val_labels = preprocess(data_dir, ann_path, begin_rate=0.7, end_rate=0.9, imgShape=imgShape)

            # train_labels = F.one_hot(torch.tensor(train_labels), num_classes=3)
            # val_labels = F.one_hot(torch.tensor(val_labels), num_classes=3)

            train_loader = DataLoader(
                dataset=TrafficLightDataset(train_data, train_labels),
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                num_workers=n_workers,
            )
            val_loader = DataLoader(
                dataset=TrafficLightDataset(val_data, val_labels),
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                num_workers=n_workers,
            )
            input_dim = 3

        return train_loader, val_loader, input_dim


def main():
    train_path = 'Data/train_dataset_128.pickle'
    val_path = 'Data/val_dataset_128.pickle'
    test_path = 'Data/test_dataset_128.pickle'

    train_loader = get_dataloader(datasetPath=train_path)
    aa = 0


if __name__ == '__main__':
    main()
