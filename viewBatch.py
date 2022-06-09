import pickle
import random

import matplotlib.pyplot as plt


file = "data_batch_1"  # 数据集
with open('./data/cifar-10-batches-py/' + file, 'rb') as f:
    dict = pickle.load(f, encoding='bytes')

print('----------%s的基本信息-------------' % file)
print('batch文件的数据类型：', type(dict))
print('字典的key：', dict.keys())
print('data的数据类型', type(dict[b'data']))
print('data的数据形状', dict[b'data'].shape)  # 输出 (10000, 3072) 说明有 10000 个样本, 3072个特征

print('batch_label: ', dict[b'batch_label'])
print('labels: ', dict[b'labels'])
print('data: ', dict[b'data'])
print('filenames: ', dict[b'filenames'])

# 随机打印10张图片
for index in range(10):
    index = random.randint(1, 1000)
    print('-----------第%d张图片信息----------' % index)
    # print('batch_label:', dict[b'batch_label'][index])
    label=dict[b'labels'][index]
    print('labels:', dict[b'labels'][index])
    print('filenames:', dict[b'filenames'][index])
    image_arr = dict[b'data'][index]  # 拿出 第 index 个样本
    image_arr = image_arr.reshape((3, 32, 32))  # 将一维向量改变形状得到这样一个元组:(高,宽,通道数)
    image_arr = image_arr.transpose((1, 2, 0))
    plt.imshow(image_arr)  # 输出图片
    plt.savefig("./viewBatch/%d.png" % index)  # 保存图片
    plt.show()
