import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageColor
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(18750 * 1, 10000), #умножаем на размер бача
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(5000, 2500),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.linear4 = nn.Sequential(
            nn.Linear(2500, 1000),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.linear5 = nn.Sequential(
            nn.Linear(1000, 100),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.linear6 = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU()
        )

        self.linear7 = nn.Sequential(
            nn.Linear(10, 4),

        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(-1)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        output = self.linear5(output)
        output = self.linear6(output)
        output = self.linear7(output)
        return output


model = ConvNet()
#model path
model.load_state_dict(torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project\conv_net_model.ckpt'))
model.eval()
#path to image
test_img = Image.open("")
test_img = transform(test_img)
test_img.resize_(1, 3, 200, 200)
coord = model(test_img)
print(coord)
coord = coord.detach().numpy()
coord[2] = coord[0] + coord[2]
coord[3] = coord[1] + coord[3]
#same path to image
out_img = Image.open("")
draw_img = ImageDraw.Draw(out_img)
draw_img.rectangle(coord, fill=None, outline='#ff0000', width=0)
out_img.show()
