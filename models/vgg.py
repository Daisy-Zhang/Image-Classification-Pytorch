import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_class = 2):
        super().__init__()

        self.conv1a = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.relu1a = nn.ReLU(inplace = True)
        self.conv1b = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.relu1b = nn.ReLU(inplace = True)
        self.maxpooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2a = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.relu2a = nn.ReLU(inplace = True)
        self.conv2b = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.relu2b = nn.ReLU(inplace = True)
        self.maxpooling2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3a = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.relu3a = nn.ReLU(inplace = True)
        self.conv3b = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.relu3b = nn.ReLU(inplace = True)
        self.conv3c = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.relu3c = nn.ReLU(inplace = True)
        self.maxpooling3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv4a = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.relu4a = nn.ReLU(inplace = True)
        self.conv4b = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.relu4b = nn.ReLU(inplace = True)
        self.conv4c = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.relu4c = nn.ReLU(inplace = True)
        self.maxpooling4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv5a = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.relu5a = nn.ReLU(inplace = True)
        self.conv5b = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.relu5b = nn.ReLU(inplace = True)
        self.conv5c = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.relu5c = nn.ReLU(inplace = True)
        self.maxpooling5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.relu1a(self.conv1a(x))
        x = self.relu1b(self.conv1b(x))
        x = self.maxpooling1(x)

        x = self.relu2a(self.conv2a(x))
        x = self.relu2b(self.conv2b(x))
        x = self.maxpooling2(x)

        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.relu3c(self.conv3c(x))
        x = self.maxpooling3(x)

        x = self.relu4a(self.conv4a(x))
        x = self.relu4b(self.conv4b(x))
        x = self.relu4c(self.conv4c(x))
        x = self.maxpooling4(x)

        x = self.relu5a(self.conv5a(x))
        x = self.relu5b(self.conv5b(x))
        x = self.relu5c(self.conv5c(x))
        x = self.maxpooling5(x)

        x = x.view(x.size()[0], -1)
        output = self.classifier(x)

        return output

def vgg16():
    return VGG16()
