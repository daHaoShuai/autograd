from nn.data import Mnist
from nn.layers import Linear, Relu
from nn.nn import NeuralNet
from nn.optim import Adam
from nn.loss import CrossEntropyLoss
from nn.train import train

# 获取mnist数据集
mnist = Mnist()
(train_data, train_label), (test_data, test_label) = mnist.load()
# 构建神经网络
net = NeuralNet([
    Linear(28 * 28, 1000),
    Relu(),
    Linear(1000, 10)
])
# 训练
train(net, train_data, train_label, num_epochs=200,
      loss=CrossEntropyLoss(), optimizer=Adam())
# 预测
print(net(test_data[0]))
