import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module): #该类继承自torch.nn.Module

    def __init__(self):
        super().__init__() #初始化方法
        self.fc1 = torch.nn.Linear(28*28, 64) #输入为28*28尺寸的MNIST图像
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10) #四个全连接层，输出为10个数字类别，该神经网络有四层

    def forward(self, x): #定义前向传播过程，x是图像输入
        x = torch.nn.functional.relu(self.fc1(x)) #每层传播中先做全链接线性计算(self.fc1(x)),再套上一个激活函数
        x = torch.nn.functional.relu(self.fc2(x)) #relu是一种激活函数
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim = 1) #输出层，通过softmax归一化，log_softmax是为了提高计算的稳定性
        return x


def get_data_loader(is_train): #用来导入数据
    to_tensor = transforms.Compose([transforms.ToTensor()]) #定义数据转换类型to_tensor（张量）
    data_set = MNIST("", is_train, transform = to_tensor, download = True) #下载MNIST数据集
    # MNIST(下载目录（空表示当前目录), 指定用于导进训练集还是测试集, transform是数据类型转换操作， download = True表示如果数据集不存在则会下载
    return DataLoader(data_set, batch_size = 15, shuffle = True)
    #一个批次包含15张图片，shuffle = true表示数据是随机打乱的，data_set是要加载的数据集
    #返回一个数据加载器


def evaluate(test_data, net): #用来评估神经网络的识别正确率
    n_correct = 0 #初始化正确预测的数量
    n_total = 0 #初始化总样本数量
    with torch.no_grad(): #上下文管理器，表示在评估过程不需要计算提督，从而节省内存和加速计算
        for (x, y) in test_data: #从测试集中按批次取出数据
            outputs = net.forward(x.view(-1, 28*28)) #计算神经网络的预测值，x.view(-1, 28*28)将输入数据展平成一个平面
            for i, output in enumerate(outputs):
            #enumerate() 函数接受一个可迭代对象作为参数，返回一个枚举对象，其中每个元素是一个包含两个值的元组，分别为索引和对应的元素值
                if torch.argmax(output) == y[i]: #argmax函数计算一个序列中最大值的序号，也就是预测的手写结果
                    n_correct += 1
                n_total += 1 #对批次中的结果进行比较，累加正确预测的数量
    return n_correct / n_total #返回正确率


def main():

    train_data = get_data_loader(is_train = True) #导入训练集
    test_data = get_data_loader(is_train = False) #导入测试集
    net = Net()

    print("initial accuracy:", evaluate(test_data, net)) #在训练开始前打印初始网络的正确率
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001) #使用 Adam 优化器对模型的参数进行优化，学习率为 0.001
    for epoch in range(2): #在同一个训练集上训练一个轮次是一个epoch
        for (x, y) in train_data:
            net.zero_grad() #初始化
            output = net.forward(x.view(-1, 28*28)) #正向传播
            loss = torch.nn.functional.nll_loss(output, y) #计算差值，nll_loss对数损失函数，匹配log_softmax中的对数运算
            loss.backward() #反向误差传播
            optimizer.step() #优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net)) #每个轮次结束后打印当前网络的正确率

    for (n, (x, _)) in enumerate(test_data): #随机抽取三张图像显示网络的预测结果
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__": #确保main函数在作为脚本直接运行时被调用
    main()