import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
import matplotlib.pyplot as plt
from ConvNet import ConvNet


#constant block
num_epochs = 10
batch_size = 1
lr = 0.00001
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


def criterion(a, b):
    overlap_area = get_overlap_area(a, b)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union_area = area_a + area_b - overlap_area

    return 1 - overlap_area / union_area


def get_counter_value(a, b):
    overlap_area = get_overlap_area(a, b)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union_area = area_a + area_b - overlap_area

    if (overlap_area != 0) and (overlap_area / union_area > 0.7):
        return 1
    else:
        return 0


def labels_resize(a, width, height):
    a[0] = a[0] / width
    a[1] = a[1] / height
    a[2] = a[2] / width
    a[3] = a[3] / height
    return a

#dataset class
class FaceDataset(Dataset):
    def __init__(self, transform, file):
      self.transform = transform
      self.labels = []
      self.images = []
      self.file = file
      #path to labels file
      i = 0
      for line in file:
        #i - number of images for dataset
        if i >= 1:
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
              l = labels_resize(l, width, height)
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


#magic
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()
start_time = time.time()
loss_list = []
acc_list = []
total = 0
correct = 0
for epoch in range(num_epochs):
    train_loss = 0
    i = 0
    if (epoch + 1) % 75 == 0:
      lr /= 2
      optimizer.param_groups[0]['lr'] = lr
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        labels.resize_([4])
        loss = criterion(output, labels.float())
        loss_list.append(loss.item())

        i += 1

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        total += 1
        correct += get_counter_value(output.detach().numpy(), np.reshape(labels.detach().numpy(), 4))
        acc_list.append(correct / total)

    print('epoch %d, loss %.4f' % (epoch + 1, train_loss / i))
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

    print('Test Accuracy of the model on', test_loader.__len__(), 'test images: {} %'.format(correct / total * 100))
fig, axes = plt.subplots(1, 2)
axes[0].plot(loss_list)
axes[0].set_title('loss')
axes[1].plot(acc_list)
axes[1].set_title('correct / total')
plt.annotate(s='accuracy on test loader: {} %'.format(correct / total * 100), xy=(0, -0.01))
plt.show()
