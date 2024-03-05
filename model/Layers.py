import numpy as np
from mpi4py import MPI
from .memory_profiler import MemoryProfiler
from .func_impl import *

np.random.seed(1)


class Layer(object):
    def __init__(self, name: str):
        self.name = name


class FCLayer(Layer):
    def __init__(
        self,
        comm,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        dp_size: int = 1,
        mp_size: int = 1,
        megatron_mp: bool = False,
        is_fc1: bool = True,
    ):
        """Defines a Fully-Connected (FC) Linear layer

        :param comm: The MPI communicator wrapper
        :type comm: object
        :param in_dim: The data dimension
        :type in_dim: int
        :param out_dim: The output dimension
        :type out_dim: int
        :param bias: Whether the layer has a bias term, defaults to False
        :type bias: bool, optional
        :param dp_size: Data parallelism size, defaults to 1
        :type dp_size: int, optional
        :param mp_size: Model parallelism size, defaults to 1
        :type mp_size: int, optional
        :param megatron_mp: Whether to perform model parallelism according to the Megatron style, defaults to False
        :type megatron_mp: bool, optional
        :param is_fc1: Whether this is the fc1 layer (True) or fc2 layer (False) in a MLP block. This info is useful only when megatron_mp==True, defaults to True
        :type is_fc1: bool, optional
        """
        super().__init__(
            name=f"FCLayer--[{in_dim}, {out_dim}]-MP Size-[{mp_size}]-MegatronP:{megatron_mp}"
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.comm = comm
        self.dp_size = dp_size
        self.mp_size = mp_size
        self.megatron_mp = megatron_mp
        self.is_fc1 = is_fc1
        self.rank = comm.Get_rank()
        self.f_peak_memory_usage = MemoryProfiler()
        self.b_peak_memory_usage = MemoryProfiler()

        (
            self.mp_group_idx,
            self.dp_group_idx,
            self.mp_comm,
            self.dp_comm,
            self.part_in_dim,
            self.part_out_dim,
        ) = get_info(
            comm=self.comm,
            rank=self.rank,
            mp_size=self.mp_size,
            dp_size=self.dp_size,
            is_fc1=self.is_fc1,
            is_megatron_mp=self.megatron_mp,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
        )

        # Initialize Weight and Bias
        self.x = None
        np.random.seed(self.rank % mp_size)
        self.w = np.random.randn(self.part_in_dim, self.part_out_dim) / np.sqrt(in_dim)
        if bias:
            self.b = np.random.randn(1, self.part_out_dim) / np.sqrt(out_dim)

        self.grad_w = np.empty_like(self.w)
        self.grad_b = np.empty_like(self.b)

        if self.megatron_mp:
            self.forward = self.megatron_forward
            self.backward = self.megatron_backward
        else:
            self.forward = self.naive_forward
            self.backward = self.naive_backward

    def naive_forward(self, x):
        """

        :param x: np.array of shape (batch_size, in_dim)
        :return: np.array of shape (batch_size, out_dim)
        """
        assert self.bias

        self.f_peak_memory_usage.start()

        if not self.is_fc1:
            x_ = naive_collect_forward_input(
                x=x,
                mp_comm=self.mp_comm,
                mp_size=self.mp_size,
            )

            x = x_

            self.x = x
            out = x @ self.w
            out = out + np.broadcast_to(self.b, out.shape)

            out_ = naive_collect_forward_output(
                out=out,
                mp_comm=self.mp_comm,
                mp_size=self.mp_size,
            )

            out = out_

            self.f_peak_memory_usage.add_var(x)
            self.f_peak_memory_usage.add_var(x_)
            self.f_peak_memory_usage.add_var(out)
            self.f_peak_memory_usage.add_var(out_)
            self.f_peak_memory_usage.add_var(self.w)
            self.f_peak_memory_usage.add_var(self.b)

            self.f_peak_memory_usage.end()
        else:
            self.x = x
            out = x @ self.w
            out = out + np.broadcast_to(self.b, out.shape)

            self.f_peak_memory_usage.add_var(x)
            self.f_peak_memory_usage.add_var(out)
            self.f_peak_memory_usage.add_var(self.w)
            self.f_peak_memory_usage.add_var(self.b)

            self.f_peak_memory_usage.end()

        return out

    def megatron_forward(self, x):
        assert self.bias
        self.f_peak_memory_usage.start()

        if not self.is_fc1:
            x = megatron_collect_forward_input(
                x=x,
                mp_comm=self.mp_comm,
                mp_size=self.mp_size,
            )

        self.x = x
        out = x @ self.w
        out = out + np.broadcast_to(self.b, out.shape)

        self.f_peak_memory_usage.add_var(x)
        self.f_peak_memory_usage.add_var(out)
        self.f_peak_memory_usage.add_var(self.w)
        self.f_peak_memory_usage.add_var(self.b)

        if not self.is_fc1:
            out_ = megatron_collect_forward_output(
                out=out,
                mp_comm=self.mp_comm,
                mp_size=self.mp_size,
            )

            out = out_

            self.f_peak_memory_usage.add_var(out_)

        self.f_peak_memory_usage.end()

        return out

    def naive_backward(self, output_grad):
        """

        :param output_grad: np.array of shape (batch_size, out_dim/mp_size)
        :return: update w, b gradients inplace and return grad_x
        """
        self.b_peak_memory_usage.start()
        self.b_peak_memory_usage.add_var(output_grad)

        if not self.is_fc1:
            # output_grad = np.split(output_grad, self.mp_size, axis=1)[self.mp_group_idx]

            output_grad = naive_collect_backward_output(
                output_grad=output_grad,
                mp_group_idx=self.mp_group_idx,
                mp_size=self.mp_size,
            )

            self.grad_b = np.sum(output_grad, axis=0, keepdims=True)
            self.grad_w = self.x.T @ output_grad

            self.b_peak_memory_usage.add_var(self.grad_b)
            self.b_peak_memory_usage.add_var(self.grad_w)

            if self.is_fc1:  ### for the first FC layer we don't need to return grad_x
                self.b_peak_memory_usage.end()
                return [None]

            grad_x = output_grad @ self.w.T

            grad_x_ = naive_collect_backward_x(
                grad_x=grad_x,
                mp_comm=self.mp_comm,
                mp_size=self.mp_size,
            )

            grad_x = grad_x_

            self.b_peak_memory_usage.add_var(grad_x)
            self.b_peak_memory_usage.add_var(grad_x_)
        else:
            self.grad_b = np.sum(output_grad, axis=0, keepdims=True)
            self.grad_w = self.x.T @ output_grad

            self.b_peak_memory_usage.add_var(self.grad_b)
            self.b_peak_memory_usage.add_var(self.grad_w)

            if self.is_fc1:  ### for the first FC layer we don't need to return grad_x
                self.b_peak_memory_usage.end()
                return [None]

            grad_x = output_grad @ self.w.T
            self.b_peak_memory_usage.add_var(grad_x)

        self.b_peak_memory_usage.end()
        return [grad_x]

    def megatron_backward(self, output_grad):
        """

        :param output_grad: np.array of shape (batch_size, out_dim/mp_size)
        :return: update w, b gradients inplace and return grad_x
        """
        self.b_peak_memory_usage.start()
        self.b_peak_memory_usage.add_var(output_grad)

        output_grad = megatron_collect_backward_output(
            output_grad=output_grad,
            mp_group_idx=self.mp_group_idx,
            mp_size=self.mp_size,
        )

        self.grad_b = np.sum(output_grad, axis=0, keepdims=True)
        self.grad_w = self.x.T @ output_grad

        self.b_peak_memory_usage.add_var(self.grad_b)
        self.b_peak_memory_usage.add_var(self.grad_w)

        if self.is_fc1:  ### for the first FC layer we don't need to return grad_x
            self.b_peak_memory_usage.end()
            return [None]

        grad_x = output_grad @ self.w.T

        grad_x = megatron_collect_backward_x(
            grad_x=grad_x, mp_size=self.mp_size, mp_comm=self.mp_comm
        )

        self.b_peak_memory_usage.add_var(grad_x)
        self.b_peak_memory_usage.end()

        return [grad_x]

    def update_weight(self, lr):
        grad_w, grad_b = collect_weight_grad(
            grad_w=self.grad_w,
            grad_b=self.grad_b,
            dp_comm=self.dp_comm,
        )

        self.grad_w = grad_w
        self.grad_b = grad_b

        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

    def zero_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)


class ReLULayer(Layer):
    def __init__(self):
        super().__init__(name=f"ReLULayer")
        self.x = None

    def forward(self, x):
        """

        :param x: np.array of shape (batch_size, in_dim)
        :return: np.array of shape (batch_size, in_dim)
        """
        self.x = x
        out = x * (x > 0)
        return out

    def backward(self, output_grad):
        """

        :param output_grad: np.array of shape (batch_size, in_dim)
        :return: return grad_x
        """
        grad_x = output_grad * (1.0 * (self.x >= 0))
        return [grad_x]


class CrossEntropyLossLayer(Layer):
    def __init__(self):
        super().__init__(name=f"CrossEntropyLossLayer")

        self.x = None
        self.y = None
        self.s = None

    def forward(self, x, y_one_hot):
        """

        :param x: np.array of shape (batch_size, num_class)
        :param y_one_hot: np.array of shape (batch_size, num_class)
        :return: loss
        """
        self.x = x
        self.y_one_hot = y_one_hot
        out = x - np.max(x, axis=1, keepdims=True)

        s = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
        self.s = s

        out = np.mean(
            np.log(np.sum(np.exp(out), axis=1)) - np.sum(out * y_one_hot, axis=1),
            axis=0,
        )
        return out

    def backward(self, output_grad=1):
        """
        :param output_grad: np.array of shape (1, )
        :return: return grad_x
        """
        grad_x = self.s - self.y_one_hot
        return [grad_x]
