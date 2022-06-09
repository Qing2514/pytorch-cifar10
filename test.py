import torchvision
from torch import nn
import torch
from PIL import Image


# 把这个模型拿过来 防止模型加载的时候报错
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


images = ['image/0-plane.png',
          'image/0-plane2.png',
          'image/0-plane3.png',
          'image/1-automobile.png',
          'image/1-automobile2.png',
          'image/1-automobile3.png',
          'image/2-bird.png',
          'image/2-bird2.png',
          'image/2-bird3.png',
          'image/3-cat.png',
          'image/3-cat2.png',
          'image/3-cat3.png',
          'image/4-deer.png',
          'image/4-deer2.png',
          'image/4-deer3.png',
          'image/5-dog.png',
          'image/5-dog2.png',
          'image/5-dog3.png',
          'image/6-frog.png',
          'image/6-frog2.png',
          'image/6-frog3.png',
          'image/7-horse.png',
          'image/7-horse2.png',
          'image/7-horse3.png',
          'image/8-ship.png',
          'image/8-ship2.png',
          'image/8-ship3.png',
          'image/9-truck.png',
          'image/9-truck2.png',
          'image/9-truck3.png']


for i in range(len(images)):
    #  原图片是ARGB类型，四通道，而我们的训练模式都是三通道的，所以这里转换成RGB三通道的格式
    image = Image.open(images[i])
    image = image.convert('RGB')

    # 使用Compose组合改变数据类型,先变成32*32的 然后在变成tensor类型
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])
    image = transform(image)
    image = torch.reshape(image, (1, 3, 32, 32))

    model = torch.load('./model/model.pth')
    model.eval()
    with torch.no_grad():
        image = image
        output = model(image)

    print(images[i], ': ', output.argmax(1))
