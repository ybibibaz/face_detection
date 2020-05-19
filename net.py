import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


def labels_resize_back(a, width, height):
    a[0] = round(a[0] * width)
    a[1] = round(a[1] * height)
    a[2] = round(a[2] * width)
    a[3] = round(a[3] * height)
    return a


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(30, 64, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(-1)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        return output


model = ConvNet()
#model path
model.load_state_dict(torch.load('D:/soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project/10_10000.ckpt'))
model.eval()
#path to image
test_img = Image.open("D:\Загрузки\убермем/Мишаня.jpg")
width, height = test_img.size
test_img = transform(test_img)
test_img.resize_(1, 3, 200, 200)
coord = model(test_img)
coord = coord.detach().numpy()
coord = labels_resize_back(coord, width, height)
#same path to image
out_img = Image.open("D:\Загрузки\убермем/Мишаня.jpg")
draw_img = ImageDraw.Draw(out_img)
draw_img.rectangle(coord, fill=None, outline='#ff0000', width=0)
out_img.show()
