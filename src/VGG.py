import torch
import torch.nn as nn

VGG_DICT = {
    11: [(64, 1), (128, 1), (256, 2), *[(512, 2)] * 2],
    16: [(64, 2), (128, 2), (256, 3), *[(512, 3)] * 2],
    19: [(64, 2), (128, 2), (256, 4), *[(512, 4)] * 2],
}


class VGG(nn.Module):
    def __init__(self, vgg_idx: int, input_dim=3) -> None:
        super(VGG, self).__init__()

        self.cnn_cfg_ls = VGG_DICT[vgg_idx]
        self.model_structure = nn.ModuleDict()
        input_channel = input_dim
        for i, (channel, layers) in enumerate(self.cnn_cfg_ls):
            module_name = f'conv{i}'
            layer_ls = []
            for _ in range(layers):
                layer_ls.extend(
                    [
                        nn.Conv2d(input_channel, channel, kernel_size=(3, 3), padding=1),
                        nn.ReLU(),
                    ]
                )
                input_channel = channel
            layer_ls.append(nn.MaxPool2d(2, 2))
            self.model_structure[module_name] = nn.Sequential(*layer_ls)

        self.full = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Dropout(),
            nn.Linear(1000, 3),
            nn.Softmax(),
        )

    def forward(self, x: torch.Tensor):
        for i in range(len(self.cnn_cfg_ls)):
            x = self.model_structure[f'conv{i}'](x)

        return self.full(x.flatten(start_dim=1))


if __name__ == '__main__':
    import numpy as np

    vgg16 = VGG(16)
    print(vgg16.model_structure)

    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    test_img = test_img.reshape((1, 3, 224, 224))
    test_tensor = torch.from_numpy(test_img).type(torch.float32)
    out = vgg16(test_tensor)

    print(VGG_DICT)
    print(VGG_DICT[16])
