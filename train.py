import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # 对RGB图片而言，数据范围是[0-255]，将数据归一化到[0,1]（是将数据除以255）
    transforms.ToTensor(),
    # 对数据按通道进行标准化，即减去均值，再除以方差，可以加快模型的收敛速度
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform_train, download=True)
test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transform_test, download=True)

print("训练集的长度:{}".format(len(train_data)))
print("测试集的长度:{}".format(len(test_data)))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(  # 顺序模型
            nn.Conv2d(3, 32, 5, 1, 2),  # 2d卷积，输入通道数3，输出通道数为32，卷积核为5x5大小
            nn.MaxPool2d(2),    # 最大池化层
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),  # 全连接层，必须将[16，5 ，5]先view()成16 * 5 * 5才能使用全连接层
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
model = Model()

# 添加tensorboard可视化数据
writer = SummaryWriter('logs')

# 损失函数：多分类用的交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器：随机梯度下降优化器（是包含动量部分的）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

i = 1  # 记录次数，用于绘制tensorboard测试集的横坐标
best_acc = 0  # 记录最佳正确率

# 开始循环训练
for epoch in range(300):
    print('开始第{}轮训练'.format(epoch))
    model.train()
    for data in train_dataloader:
        # 读入数据，数据分开，一个是图片数据，一个是真实值
        imgs, targets = data
        imgs = imgs
        targets = targets
        output = model(imgs)  # 前向传播，拿到预测值
        loss_in = loss(output, targets)  # 计算损失值
        optimizer.zero_grad()  # 优化开始，先梯度清零
        loss_in.backward()  # 梯度回传
        optimizer.step()  # 更新梯度

    sum_loss = 0  # 记录总体损失值

    # 每轮训练完成跑一下测试数据看看情况
    accurate = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            # 这里的每一次循环 都是一个 minibatch  一次for循环里面有64个数据。
            imgs, targets = data
            imgs = imgs
            targets = targets
            output = model(imgs)
            loss_in = loss(output, targets)

            # print('output: ', output)
            sum_loss += loss_in
            accurate += (output.argmax(1) == targets).sum()

    acc = accurate / len(test_data) * 100
    print('第{}轮测试集的正确率:{:.2f}%'.format(epoch, acc))
    print('第{}轮测试集的总损失率:{:.2f}%'.format(epoch, sum_loss))
    writer.add_scalar('acc', acc, i)
    writer.add_scalar('sum loss', sum_loss, i)
    i += 1

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'acc': best_acc,
            'sum_loss': sum_loss,
            'model': model.state_dict(),
        }, './model/best_model.pth')
        torch.save(model, './model/model.pth')
        print("第{}轮模型训练数据已保存".format(epoch))

writer.close()

# tensorboard可视化
# tensorboard --logdir="logs"
