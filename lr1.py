import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
import cv2

#constant block
num_epochs = 5
batch_size = 1
lr = 0.02
#function for image shaping
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


#dataset class
class FaceDataset(Dataset):
    def __init__(self, transform):
      self.transform = transform
      self.labels = []
      self.images = []
      #path to labels file
      f = open("D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project\datoset\list_bbox_celeba.txt", "r")
      i = 0
      for line in f:
        #i - number of images for dataset
        if(i > 30):
          break
        l = line.replace('\n', '').split(' ')
        filename = l[0]
        try:
              #path to images folder
              with Image.open("D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/project/datoset/img_celeba/" + filename) as image:
                  img = self.transform(image.copy())
                  self.images.append(img)
              l = list(map(int, l[1::]))
              l[2] = l[0] + l[2]
              l[3] = l[1] + l[3]
              self.labels.append(np.array(l))
              i += 1
        except:
          print('Could not load image:', filename)
          break
      f.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]
        Y = self.labels[idx]
        return X, Y

#dataset creation
dataset = FaceDataset(transform)
data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)


#nn class
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
            nn.Linear(18750 * 1, 10000),
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


#magic
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
start_time = time.time()
for epoch in range(num_epochs):
    train_loss = 0
    i = 0
    if(epoch % 20 == 0):
      lr /= 5
      optimizer.param_groups[0]['lr'] = lr
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(y, output)
        del(x)
        del(y)
        del(output)
        i += 1
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('epoch %d, loss %.4f' % (epoch, train_loss / i))
    print('current learning time: ' + str(time.time() - start_time))


print('learning time: ' + str(time.time() - start_time))
#path where to save model
torch.save(model.state_dict(), 'D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/project/' + 'conv_net_model.ckpt')