import time
import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# set device
device = torch.device('cuda')


class Model(nn.Module):
    """
    Define a valid model
    """

    def __init__(self):
        super(Model, self).__init__()
        self.input = nn.Linear(4, 10)
        self.h1 = nn.Linear(10, 100)
        self.h2 = nn.Linear(100, 80)
        self.h3 = nn.Linear(80, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        out = self.input(x)
        out = torch.relu(out)
        out = self.h1(out)
        out = torch.relu(out)
        out = self.h2(out)
        out = torch.relu(out)
        out = self.h3(out)
        out = torch.relu(out)
        out = self.output(out)
        return out


def train(lr, epochs, train_data, val_data, weight_path):
    start_time = time.time()

    model = Model()
    model.to(device)

    # set optimizer for model
    optimizer = optim.Adam(model.parameters(), lr)  # 随机梯度下降法
    criterion = nn.MSELoss()  # 均方差损失函数

    # set min loss and save path
    min_loss = 0.5
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        loses = []

        for _, (train_input, train_output) in enumerate(train_data):
            output = model.forward(train_input)  # 计算模型输出结果
            loss = criterion(output, train_output)  # 损失函数
            loses.append(loss.item())
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度 后向传递过程
            optimizer.step()  # 优化权重与偏差矩阵

        # calculate average train loss
        train_loss = sum(loses) / len(loses)
        train_losses.append(train_loss)

        logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
        target = []
        model.eval()  # 启动测试模式

        for data, targets in val_data:  # 输出验证集的平均误差
            logits = model.forward(data).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            logit.append(logits[0])
            target.append(targets[0])

        # calculate val loss
        val_loss = criterion(torch.tensor(logit), torch.tensor(target)).detach().cpu().numpy().tolist()
        val_losses.append(val_loss)

        # if epoch % 100 == 0:
        print(str(epoch) + "/" + str(epochs), "train_loss", train_loss,
              "val_loss", val_loss, sep='\t')

        if val_loss < min_loss:
            min_loss = val_loss
            print("Model has been saved!")
            torch.save(model.state_dict(), weight_path)

        print('----')

    end_time = time.time()
    print("Time usage is", (end_time - start_time))

    return train_losses, val_losses


def load_model(weight_file):
    model = Model()
    model.load_state_dict(torch.load(weight_file))
    model.to(device)
    print("Model has been loaded!")
    return model


def test(test_data, trained_model):

    start_time = time.time()

    with torch.no_grad():
        distances = []
        for data, targets in test_data:
            outputs = trained_model(data)
            outputs = outputs.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            distance = math.sqrt(math.pow((outputs[0] - targets[0]) * 808, 2) + math.pow(
                (outputs[1] - targets[1]) * 448, 2))
            distances.append(distance)

        avg_result = sum(distances) / len(distances)

        print("Average Distance Deviation is:", avg_result, sep='\t')

    end_time = time.time()
    print("Time usage is", (end_time - start_time))
