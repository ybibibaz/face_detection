import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from ConvNet import ConvNet


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
