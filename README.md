# dl_study

Todo:

用leakerelu作为激活函数看看效果

不同函数的拟合，比如幂函数

看下学习过程中权重变化的趋势，已经抓了2+2+1忘了的raw data，等待研究。

√ 做了一次heartrate，发现每个batch都会调用一次MyDataset::__getitem__()来获取数据，这个函数是在CPU上运行的，优化这个函数有利于减少CPU占用。当GPU并行度足够高时，这个函数可能成为瓶颈。
√ 改用numpy替代list来加载数据，简单测试了下，效率提升了3倍。用list加载，训练10轮用时42s的模型，用numpy加载，10轮只要13s。

接着做 增加随机噪声.xlsx
√ 添加了几个维度噪声，学习效果时候差不多
√ 做一次全噪声的训练，看看有什么结果，结果也能误差也能降到30%，打印学完的权重，是每个维度上都会取一点，可见误差大于30%就是未学成

√ 偶尔有收敛误差较大，可能是参数没有初始化的原因，初始化参数用下面的参数暂时未发现学废的情况。
num_of_neures_in_input_layer = 3
num_of_neures_in_hidden_layer = 5
activate_fun = nn.ReLU
def criterion_fun():
    return nn.MSELoss()
    # return MAELoss()
optimizer_fun = optim.Adam
batch_size = 50
lr = 0.00001
epic = 1600

√ 添加两个神经元作为倒数第二层.xlsx
不知道为什么会是折线，理论上应该是条直线。
画下当倒数第二层x1==0时，画下x2，y的图像，应该是一条直线

