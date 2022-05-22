import sys, os, json
from typing import List, Tuple

from tqdm import tqdm
import cv2
import numpy as np

# sys.path.append(os.path.abspath(__package__))
sys.path.append(os.path.abspath('./'))
# print(sys.path)
from submodules.FileTools.WordOperator import str_format
from submodules.FileTools.PickleOperator import save_pickle


def load_ann_json(path: str) -> List[tuple]:
    with open(path, 'r') as f:
        anns_dict: dict = json.load(f)

    data = []
    for ann in anns_dict['annotations']:
        ann: dict
        filename = ann['filename'].split('\\')[-1]

        # * bbox = [x_min, y_min, x_max, y_max]
        bbox = [value for value in ann['bndbox'].values()]
        try:
            color = ann['inbox'][0]['color']
        except IndexError:
            continue

        # print(filename)
        # print(bbox)
        # print(color)
        data.append((filename, bbox, color))

    return data


def check_label(ann_list: list):
    label_set = set()
    for ann in ann_list:
        label_set.add(ann[-1])

    # print(label_set)

    return sorted(list(label_set))


def load_imgs(data_dir: str, data: list, imgShape: Tuple[int] = None):
    imgs = []
    pbar = tqdm(data)
    for ann in pbar:
        # print(f'{data_dir}/{ann[0]}')
        img = cv2.imread(f'{data_dir}/{ann[0]}')
        h_min, w_min, h_max, w_max = [int(coord) for coord in ann[1]]
        img = img[w_min:w_max, h_min:h_max]
        # cv2.imwrite(f'./out/test/{ann[0]}', img)
        if imgShape is not None:
            img = cv2.resize(img, imgShape)
        imgs.append(img)

    return imgs


def generate_clsIDs(cls: list):
    cls_dict = {}
    for i, cls_key in enumerate(cls):
        cls_dict[cls_key] = i

    return cls_dict


# def create_dataset(data_infos: List[tuple], imgs: List[np.ndarray], cls_dict: dict):
#     dataset = []

#     for img, data_info in zip(imgs, data_infos):
#         data_info: tuple
#         label_id = cls_dict[data_info[-1]]

#         dataset.append((img, label_id))

#     return dataset


def preprocess(
    data_dir: str,
    ann_path: str,
    begin_rate: float = 0.0,
    end_rate: float = 0.7,
    imgShape: Tuple[int] = (128, 128),
    savePicklePath: str = None,
    **kwargs,
):

    # filenames = get_filenames(data_dir, '*.jpg', withDirPath=False)
    # filenames = np.array(sorted([int(filename.split('.')[0]) for filename in filenames]), dtype=np.int16)
    data_infos = load_ann_json(ann_path)

    num_dataset = len(data_infos)

    num_begin, num_end = int(num_dataset * begin_rate), int(num_dataset * end_rate)

    data_infos = data_infos[num_begin:num_end]

    # print(f"Number of Total Dataset: {str_format(num_dataset, fore='y')}")
    print(
        f"Number of {kwargs['word']} Dataset: {str_format(f'{num_end-num_begin}', fore='y')}, from {num_begin}:{num_end} of Total Dataset"
    )

    imgs = load_imgs(data_dir, data_infos, imgShape)

    cls_dict = generate_clsIDs(check_label(data_infos))

    labels = [cls_dict[data_info[-1]] for data_info in data_infos]

    # dd = {'yellow': 0, 'green': 0, 'red': 0}
    # for data_info in data_infos:
    #     dd[data_info[-1]] += 1

    # print(dd)
    # print()

    if savePicklePath is not None:
        save_pickle([imgs, labels], savePicklePath)

    return imgs, labels


if __name__ == '__main__':
    data_dir = 'Data/train_dataset/train_images'
    ann_path = 'Data/train_dataset/train.json'

    preprocess(
        data_dir,
        ann_path,
        imgShape=(128, 128),
        savePicklePath='Data/train_dataset.pickle',
        **{
            'word': 'Train',
        },
    )
    preprocess(
        data_dir,
        ann_path,
        begin_rate=0.7,
        end_rate=0.9,
        imgShape=(128, 128),
        savePicklePath='Data/val_dataset.pickle',
        **{'word': 'Val'},
    )
    preprocess(
        data_dir,
        ann_path,
        begin_rate=0.9,
        end_rate=1.0,
        imgShape=(128, 128),
        savePicklePath='Data/test_dataset.pickle',
        **{'word': 'Test'},
    )
