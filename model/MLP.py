import numpy as np
from .Layers import FCLayer, ReLULayer, CrossEntropyLossLayer

np.random.seed(1)


class MLPModel(object):
    def __init__(
        self,
        comm,
        dp_size: int = 1,
        mp_size: int = 1,
        megatron_mp: bool = False,
        feature_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ):
        """Defines a MLP block

        :param comm:  The MPI communicator wrapper
        :type comm: object
        :param dp_size: Data parallelism size, defaults to 1
        :type dp_size: int, optional
        :param mp_size: Model parallelism size, defaults to 1
        :type mp_size: int, optional
        :param megatron_mp: Whether to perform model parallelism according to the Megatron style, defaults to False
        :type megatron_mp: bool, optional
        :param feature_dim: The feature dimension, defaults to 784
        :type feature_dim: int, optional
        :param hidden_dim: The hidden dimension, defaults to 256
        :type hidden_dim: int, optional
        :param output_dim: The output dimension, defaults to 10
        :type output_dim: int, optional
        """
        self.fc1 = FCLayer(
            comm=comm,
            in_dim=feature_dim,
            out_dim=hidden_dim,
            bias=True,
            dp_size=dp_size,
            mp_size=mp_size,
            megatron_mp=megatron_mp,
            is_fc1=True,
        )
        self.relu = ReLULayer()
        self.fc2 = FCLayer(
            comm=comm,
            in_dim=hidden_dim,
            out_dim=output_dim,
            bias=True,
            dp_size=dp_size,
            mp_size=mp_size,
            megatron_mp=megatron_mp,
            is_fc1=False,
        )
        self.cross_entropy_loss = CrossEntropyLossLayer()
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mp_size = mp_size
        self.dp_size = dp_size

    def forward(self, x, y):
        """

        :param x: input images of shape (batch_size, feature_dim)
        :param y: labels of shape (batch_size, )
        :return: loss
        """
        y_one_hot = np.zeros((x.shape[0], 10))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        predict = np.argmax(x, axis=1)
        acc = (y == predict).sum() / y.shape[0]
        loss = self.cross_entropy_loss.forward(x, y_one_hot)
        return loss, acc

    def backward(self):
        grad_x = self.cross_entropy_loss.backward()[0]
        grad_x = self.fc2.backward(output_grad=grad_x)[0]
        grad_x = self.relu.backward(output_grad=grad_x)[0]
        _ = self.fc1.backward(output_grad=grad_x)[0]

    def update_weights(self, lr):
        self.fc1.update_weight(lr)
        self.fc2.update_weight(lr)

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def get_rank(self):
        return self.rank
