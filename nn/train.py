from nn.tensor import Tensor
from nn.nn import NeuralNet
from nn.loss import Loss, MSE
from nn.optim import Optimizer, SGD
from nn.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()
          ) -> None:
    """
    训练方法
    net             : 神经网络
    inputs          : 输入数据
    targets         : 对应的真实数据
    num_epochs      : 训练轮数
    iterator        : 把数据处理成批数据
    loss            : 损失函数
    optimizer       : 优化器
    """
    for epoch in range(1, num_epochs+1):
        # 计算每一轮训练的损失值
        epoch_loss = 0.0
        # 获取批数据
        for batch in iterator(inputs, targets):
            # 预测
            predicted = net(batch.inputs)
            # 得到损失
            epoch_loss += loss.loss(predicted, batch.targets)
            # 得到损失的梯度
            grad = loss.grad(predicted, batch.targets)
            # 根据损失的梯度反向传播得到网络中权重的梯度
            net.backward(grad)
            # 通过梯度更新权重
            optimizer.step(net)
            # 梯度清零
            optimizer.zero_grad(net)
        print(f'epoch is {epoch}, epoch_loss is {epoch_loss}')
