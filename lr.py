import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time

#constant block
num_epochs = 5
batch_size = 1
lr = 0.001
f = open("D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project\datoset\list_bbox_celeba.txt", "r")
#function for image shaping
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


def get_overlap_area(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])

    if width > 0 and height > 0:
        return width * height
    else:
        return 0


def get_counter_value(a, b):
    overlap_area = get_overlap_area(a, b)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union_area = area_a + area_b - overlap_area

    if ((overlap_area != 0) and (overlap_area / union_area > 0.7)):
        return 1
    else:
        return 0


def labels_resize(a, width, height):
    a[0] = round(a[0] * (200 / width))
    a[1] = round(a[1] * (200 / height))
    a[2] = round(a[2] * (200 / width))
    a[3] = round(a[3] * (200 / height))


#dataset class
class FaceDataset(Dataset):
    def __init__(self, transform, file):
      self.transform = transform
      self.labels = []
      self.images = []
      self.file = file
      #path to labels file
      i = 0
      for line in f:
        #i - number of images for dataset
        if(i >= 100):
          break
        l = line.replace('\n', '').split(' ')
        filename = l[0]
        try:
              #path to images folder
              with Image.open("D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/project/datoset/img_celeba/" + filename) as image:
                  width, height = image.size
                  img = self.transform(image.copy())
                  self.images.append(img)
              l = list(map(int, l[1::]))
              l[2] = l[0] + l[2]
              l[3] = l[1] + l[3]
              labels_resize(l, width, height)
              self.labels.append(np.array(l))
              i += 1
        except:
          print('Could not load image:', filename)
          break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]
        Y = self.labels[idx]
        return X, Y


#dataset creation
train_dataset = FaceDataset(transform, f)
test_dataset = FaceDataset(transform, f)
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
f.close()


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


#magic
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
start_time = time.time()
for epoch in range(num_epochs):
    train_loss = 0
    i = 0
    if((epoch + 1) % 5 == 0):
      lr /= 2
      optimizer.param_groups[0]['lr'] = lr
    for x, y in train_loader:
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
    torch.save(model.state_dict(),
               'D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/project/' + 'conv_net_model.ckpt')
print('learning time: ' + str(time.time() - start_time))


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        total += 1
        correct += get_counter_value(outputs.detach().numpy(), np.reshape(labels.detach().numpy(), 4))

    print('Test Accuracy of the model on the', test_loader.__len__(), 'test images: {} %'.format((correct / total) * 100))
