import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import math
from datetime import datetime
import torch.nn.init as init
import sys
# from heartrate import trace
# trace(browser=True)


num_of_neures_in_input_layer = 1000
num_of_neures_in_hidden_layer = 5
activate_fun = nn.ReLU
# activate_fun = nn.Sigmoid
# activate_fun = nn.Tanh
def criterion_fun():
    return nn.MSELoss()
    # return MAELoss()
optimizer_fun = optim.Adam
# optimizer_fun = optim.Adagrad
# optimizer_fun = optim.SGD
batch_size = 50
lr = 0.00001


epic = 1000
# input_weights_file = f'all_noize(1000+5+1).ReLu.MSELoss.Adam.50.(0.00001).(1600)-N-init.1.wt'
# input_weights_file = f'({num_of_neures_in_input_layer}+{num_of_neures_in_hidden_layer}+1).ReLu.MSELoss.Adam.{batch_size}.({lr:f}).({epic})-N-init.1.wt'
# output_weights_file = f'({num_of_neures_in_input_layer}+{num_of_neures_in_hidden_layer}+1).ReLu.MSELoss.Adam.{batch_size}.({lr:f}).({epic})-N-init.1.wt'
# output_weights_file = f'all_noize(1000+5+1).ReLu.MSELoss.Adam.50.(0.00001).(1600+1200)-N-init.1.wt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

isNeedNormalize = True    # 是都对数据做归一化
# checkValidity = True    # train集 全集验证
# per_batch_val = True    # 每个batch打印一次验证结果
per_epic_val = True       # 每个epic打印一次验证结果

data_file = "G-small_data.txt"  # 数据文件路径

# torch.set_printoptions(threshold=sys.maxsize)  # 打印张量不省略
torch.set_printoptions(sci_mode=False)  # 打印张量禁用科学计数法

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predictions, targets):
        absolute_diff = torch.abs(predictions - targets)
        mae = torch.mean(absolute_diff)
        return mae

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_of_neures_in_input_layer, num_of_neures_in_hidden_layer)
        self.fc2 = nn.Linear(num_of_neures_in_hidden_layer, 1)
        # self.fc2 = nn.Linear(num_of_neures_in_hidden_layer, 2)
        # self.fc3 = nn.Linear(2, 1)
        self.activate = activate_fun()

        # 对权重进行He初始化
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        # init.kaiming_uniform_(self.fc3.weight, a=0, mode='fan_in', nonlinearity='relu')

        # 对偏差进行零初始化
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        # init.zeros_(self.fc3.bias)

    def forward(self, x):
        # print("!!!!!!x.shape:", x.shape)
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        # 打印倒数第二层神经元值
        # print(x)
        # for line in x.tolist():
        #     # for x1x2 in line:
        #     print(f'{line[0]:.5f}')
        # x = self.fc3(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.inputs, self.targets = self._load_data(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_value = self.inputs[idx]
        target_value = self.targets[idx]

        # 数据预处理
        input_value = torch.tensor(input_value)
        target_value = torch.tensor([target_value])

        return input_value, target_value

    def _load_data(self, data_file):
        # 从文件中加载数据的逻辑, 返回输入数据列表和目标数据列表
        inputs = np.array([], dtype=np.float32)
        targets = np.array([], dtype=np.float32)
        with open(data_file, 'r') as f:
            for line in f:
                # 假设每行数据包含输入和目标，使用空格分隔
                input_value, target_value = line.strip().split()
                inputs = np.append(inputs, np.float32(input_value))
                targets = np.append(targets, np.float32(target_value))

        if ("isNeedNormalize" in globals()):
            print("Normalize data.")
            inputs_min = np.min(inputs);  targets_min = np.min(targets)
            inputs = 2 * (inputs - inputs_min) / (np.max(inputs) - inputs_min) - 1
            targets = (targets - targets_min) / (np.max(targets) - targets_min)

        input_noise = np.random.uniform(-1, 1, size=(len(inputs),(num_of_neures_in_input_layer - 1))).astype(float)
        inputs = np.column_stack((inputs, input_noise))
        # inputs = np.random.uniform(-1, 1, size=(len(inputs),(num_of_neures_in_input_layer))).astype(float) # 全噪声，每次随机噪声不一样，不能接着断点续学

        print("inputs.shape:", np.array(inputs).shape)

        return inputs, targets

class ToTensor(object):
    def __call__(self, x):
        return torch.tensor(x)

def verifyResult(net):
    "成果验证"
    # test_x = [[1], [10], [100], [1000], [10000], [100000], [1000000], [10000000], [100000000], [1000000000]]
    # test_x = [[-1], [-10], [-100], [-1000], [-10000], [-100000], [-1000000], [-10000000], [-100000000], [-1000000000]]
    # test_x = [[-1], [-11], [-22], [-33], [-44], [-55], [-66], [-77], [-88], [-99]]
    # test_x = [[1], [11], [22], [33], [44], [55], [66], [77], [88], [99]]
    test_x = [[i] for i in np.arange(-1e3, 1e3, 0.1)]
    test_y = [4.9 * i[0]**2 for i in test_x]

    "归一化"
    if ("isNeedNormalize" in globals()):
        print("Normalize data.")
        np_test_x = np.array(test_x)
        max_value = np.max(np_test_x)
        min_value = np.min(np_test_x)
        test_x =  [[2 * (x[0] - min_value) / (max_value - min_value) - 1] for x in test_x]
        max_value = max(test_y)
        min_value = min(test_y)
        test_y = [(y - min_value) / (max_value - min_value) for y in test_y]

    "数据张量化"
    test_x_tensor = torch.tensor(test_x)
    test_x_tensor = test_x_tensor.to(device)
    test_x_tensor = test_x_tensor.to(torch.float32)
    test_y_tensor = torch.tensor(test_y)
    test_y_tensor = test_y_tensor.to(device)
    test_y_tensor = test_y_tensor.to(torch.float32)
    test_y_tensor = torch.tensor(test_y).reshape(-1, 1)

    output_tensor = net(test_x_tensor)
 
    # print("test_x_tensor", test_x_tensor)
    # print("test_y_tensor", test_y_tensor)
    # print("output_tensor", output_tensor)
    diff_tensor = output_tensor - test_y_tensor
    print("output_tensor - test_y_tensor", diff_tensor)
    percent_tensor = diff_tensor/test_y_tensor
    diff_tensor = diff_tensor.squeeze() # 一维化
    percent_tensor = percent_tensor.squeeze() # 一维化
    # for value in diff_tensor.tolist():
    #     print(f'{value:.5f}')

    test_loss = criterion_fun()(output_tensor, test_y_tensor)
    print("Validation Criterion Loss:", test_loss.item())

def validate_loss(val_dataloader, criterion):
    with torch.no_grad():
        val_criterion_losses = []
        val_abs_losses = []
        for val_inputs, val_targets in val_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_output_tensor = net(val_inputs)
            val_abs_loss = torch.mean(torch.abs(val_output_tensor - val_targets))
            val_abs_losses.append(val_abs_loss.item())
            val_criterion_loss = criterion(val_output_tensor, val_targets)
            val_criterion_losses.append(val_criterion_loss.item())
            # if (val_abs_loss > 0.25):
            #     print("val_output_tensor:", val_output_tensor)
            #     print("\033[31mval_targets:\033[0m", val_targets)
            #     print("torch.abs(val_output_tensor - val_targets):", torch.abs(val_output_tensor - val_targets))
            #     print("val_abs_loss:", val_abs_loss)
            #     print("val_criterion_loss:", val_criterion_loss)
        # 打印训练损失和验证损失
        # print("Train Loss:", loss.item())
        # print("Validation Abs Loss:", sum(val_abs_losses) / len(val_abs_losses))
        # print("Validation Criterion Loss:", sum(val_criterion_losses) / len(val_criterion_losses))
        # print("", sum(val_criterion_losses) / len(val_criterion_losses))
        print("\033[32m", sum(val_abs_losses) / len(val_abs_losses), "\033[0m")

def train(net):
    # 数据预处理和加载
    data_transform = transforms.Compose([
        ToTensor()  # 将数据转换为张量
        # transforms.Normalize(mean=[mean_value], std=[std_value])  # 数据归一化
    ])

    # 创建自定义数据集实例
    dataset = MyDataset(data_file, transform=data_transform)

    # 将数据集分为训练集和验证集
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% 作为训练集
    val_size = dataset_size - train_size  # 剩余 20% 作为验证集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = criterion_fun()
    optimizer = optimizer_fun(net.parameters(), lr=lr)

    before_train_time = datetime.now()
    print("prepare time:", (before_train_time - start_time))
    last_around_start_time = before_train_time

    for i in range(epic):
        # print("###################### epic:", i)
        # start_train_time = datetime.now()
        # print("epoch train time:\033[33m", (start_train_time - last_around_start_time), "\033[0m")
        # print("    total train time:\033[34m", (start_train_time - before_train_time), "\033[0m")
        # last_around_start_time = start_train_time

        # 训练每个批次
        for batch_inputs, batch_targets in train_dataloader:
            # 将批次数据移动到CUDA设备
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # print("batch_inputs.shape:", batch_inputs.shape)
            # print("batch_targets.shape:", batch_targets.shape)

            # 前向传播
            output_tensor = net(batch_inputs)
            # print("batch_inputs", (1/2) * 9.8 * (batch_inputs*100)**2)
            # print("batch_targets", batch_targets*49000)
            # print("diff", ((1/2) * 9.8 * (batch_inputs*100)**2) - batch_targets*49000)

            # 计算训练损失
            loss = criterion(output_tensor, batch_targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ("per_batch_val" in globals()):
                # 在每个batch结束后计算验证损失
                validate_loss(val_dataloader, criterion)

        if ("per_epic_val" in globals()):
            # 在每个epic结束后计算验证损失
            validate_loss(val_dataloader, criterion)

    if ("output_weights_file" in globals()):
        print("Save weights to:", output_weights_file)
        torch.save(net.state_dict(), output_weights_file)
    else:
        print("No weights saved.")

    end_time = datetime.now()
    print("time:", (end_time - start_time))

if __name__ == '__main__':
    start_time = datetime.now()
    print("start_time:", start_time)

    # 创建网络实例
    net = MyNetwork()

    # 将模型加载到CUDA设备
    net.to(device)
    if ("input_weights_file" in globals()):
        print("Load weights from:", input_weights_file)
        net.load_state_dict(torch.load(input_weights_file))
    else:
        print("No weights loaded.")

    # 遍历网络的所有参数，并打印它们的名称、形状和值
    # for name, param in net.named_parameters():
    #     print(f"Parameter name: {name}\t Parameter shape: {param.shape}")
    #     print(f"Parameter value: {param.data}\n")

    if ("checkValidity" in globals()):
        verifyResult(net)
    else:
        train(net)
