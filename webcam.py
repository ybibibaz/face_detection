import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


def labels_resize_back(a, width, height):
    a[0] = round(a[0] / (200 / width))
    a[1] = round(a[1] / (200 / height))
    a[2] = round(a[2] / (200 / width))
    a[3] = round(a[3] / (200 / height))
    return a


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
        self.conv4 = nn.Sequential(
            nn.Conv2d(30, 64, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(512, 100),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.linear4 = nn.Sequential(
            nn.Linear(100, 4),

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
        output = self.linear4(output)
        return output


model = ConvNet()
#model path
model.load_state_dict(torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project\conv_net_model.ckpt'))
model.eval()
cap = cv2.VideoCapture('test.mp4')
while(cap.isOpened()):
  ret, frame = cap.read()
  if not ret:
      raise ValueError("unable to load Image")
  img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  im_pil = Image.fromarray(img)
  width, height = im_pil.size
  im_pil = transform(im_pil)
  im_pil.resize_(1, 3, 200, 200)
  coord = model(im_pil)
  coord = coord.detach().numpy()
  coord = labels_resize_back(coord, width, height)
  frame = cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255))
  cv2.imshow('video', frame)
  if (cv2.waitKey(1) & 0xFF == ord('q')):
      break
cap.release()
cv2.destroyAllWindows()
