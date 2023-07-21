# dl_study

接着做 增加随机噪声.xlsx
做了几次效果时候差不多
做一次全噪声的训练，看看有什么结果

偶尔有收敛误差较大，可能是参数没有初始化的原因，初始化参数用下面的参数暂时未发现学废的情况。
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

添加两个神经元作为倒数第二层.xlsx
不知道为什么会是折线，理论上应该是条直线。
画下当倒数第二层x1==0时，x2，y的图像，应该是一条直线

