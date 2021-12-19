import torch
from utils import MyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = MyDataset('/home/users/matsuda/work/Datasets/COCO')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

batch_iter = iter(dataloader)

image, target = next(batch_iter)

print("size : ", image.size())
image = torch.squeeze(image)
print("size : ", image.size())

cv_img = image.detach().numpy()

# pltの形 (width, height, channel) に合わせる
cv_img = cv_img.transpose(1, 2, 0)
#print("shape : ", cv_img.shape())

# もともと GBR になってるのを RGB にする
cv_img = cv_img[:, :, (2, 1, 0)]
#print("shape : ", cv_img.shape())

plt.imshow(cv_img)
plt.show()
