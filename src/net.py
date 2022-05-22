import torch.nn as nn

from VGG import VGG


class Net_Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Net_Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 32, (3, 3), stride=1, padding=1),  # [64, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2, 0),  # [32, 64, 64]
            nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),  # [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2, 0),  # [256, 32, 32]
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),  # [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2, 0),  # [64, 16, 16]
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),  # [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 8, 8]
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)

        return self.fc(out)
