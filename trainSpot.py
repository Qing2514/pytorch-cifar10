import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# 创造数据
n_data = torch.ones(100, 2)  # 数据的的基本形态
x0 = torch.normal(2 * n_data, 1)  # class0 x shape=(100,2) 创建一个服从均值为2*n_data的张量，标准差为均为1的tensor
y0 = torch.zeros(100)  # class0 y shape=(100,1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x shape=(100,2) 创建一个服从均值为-2*n_data的张量，标准差为均为1的tensor,shape=(100,2)
y1 = torch.ones(100)  # class1 y shape=(100,1)

# 合并数据（cat是cancatenate的意思)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit float 其中0是按行拼接，1是按列拼接
y = torch.cat((y0, y1)).type(torch.LongTensor)  # LongTensor = 64-bit integer

# 画图
plt.scatter(x.data.numpy()[:, 0], x.data[:].numpy()[:, 1])
plt.show()


# 创建网络
class Net(torch.nn.Module):  # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        X = F.relu(self.hidden(x))  # 激活函数
        X = self.out(X)  # 输出值，但这不是预测值，预测值还需另外计算
        return X


# 输入是两个特征，x对应的特征和y对应的特征，输出是2个类，0和1
net = Net(n_feature=2, n_hidden=10, n_output=2)
# 输出为[0,1]说明图片为class1，若是[1,0],说明输出为class0。这是二分类
# 输出为[0,1,0]说明图片为class1，若是[1,0,0],说明输出为class0,若是[0,0,1],说明输出为class2。这是三分类
# print(net)

# 训练 输出是概率
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入net的所有参数和学习率
loss_func = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = net(x)  # 把数据x作为输入，输出分析值
    loss = loss_func(out, y)  # 计算预测值和真实值两者误差
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameters上

    # 可视化
    if i % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), dim=1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
