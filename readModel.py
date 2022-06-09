import torch
from torch import nn


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


print(Model())  # 查看网络模型结构
# content = torch.load('./model/best_model.pth')
# content = torch.load('./model/model.pth')
# content = torch.load('./checkpoint/ckpt.pth')
# print(content.keys())  # 输出模型中的key
# print(content)  # 输出模型所有数据
# 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['best_acc'])
