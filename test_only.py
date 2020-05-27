import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from ConvNet import ConvNet
import xlwt


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


pic_count = 100
book = xlwt.Workbook()
sheet1 = book.add_sheet("Sheet1")


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
    a[0] = a[0] * (200 / width) / 200
    a[1] = a[1] * (200 / height) / 200
    a[2] = a[2] * (200 / width) / 200
    a[3] = a[3] * (200 / height) / 200


#dataset class
class FaceDataset(Dataset):
    def __init__(self, transform, file):
      self.transform = transform
      self.labels = []
      self.images = []
      self.file = file
      self.names = []
      #path to labels file
      i = 0
      for line in file:
        #i - number of images for dataset
        if i >= 10000 + pic_count:
            break
        if i < 10000:
            i += 1
            continue
        l = line.replace('\n', '').split(' ')
        filename = l[0]
        try:
              #path to images folder
              with Image.open("D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/project/datoset/img_celeba/" + filename) as image:
                  width, height = image.size
                  img = self.transform(image.copy())
                  self.images.append(img)
                  self.names.append(filename)
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
test_dataset = FaceDataset(transform, f)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
f.close()


model = ConvNet()
model.load_state_dict(torch.load('D:/soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\project/10_10000.ckpt'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        row = sheet1.row(total)
        row.write(0, test_dataset.names[total])
        row.write(1, criterion(outputs.detach().numpy(), np.reshape(labels.detach().numpy(), 4)))
        total += 1
        correct += get_counter_value(outputs.detach().numpy(), np.reshape(labels.detach().numpy(), 4))

    print('Test Accuracy of the model on', test_loader.__len__(), 'test images: {} %'.format(correct / total * 100))
book.save("stuff/test.xls")
