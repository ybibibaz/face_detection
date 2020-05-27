import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
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
model.load_state_dict(torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project/10_10000.ckpt'))
model.eval()
cap = cv2.VideoCapture('test.mp4')
while cap.isOpened():
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
