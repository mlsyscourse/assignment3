import numpy as np
import h5py
from mpi4py import MPI
from mpi_wrapper import Communicator
from logger import log_args, log_stats
import argparse, os
from data.data_parallel_preprocess import split_data

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--mp_size", type=int, help="model parallel size", default=1)
parser.add_argument("--dp_size", type=int, help="data parallel size", default=1)
parser.add_argument(
    "--megatron-mp",
    action="store_true",
    help="Use this flag to enable Megatron-style model parallelism",
)

from model.MLP import MLPModel


def lr_schedule(init_lr, iter_num, decay=0.9, stage_num=100):
    return init_lr * (decay ** (np.floor(iter_num / stage_num)))


def train_mlp(
    x_train, y_train, x_test, y_test, model, num_epoch=3, batch_size=60, init_lr=0.1
):
    iter_num = 0
    num_examples = x_train.shape[0]
    rank = model.get_rank()
    for epoch in range(num_epoch):
        # Train
        if rank == 0:
            print("*" * 40 + "Training" + "*" * 40)
        for i in range(0, num_examples, batch_size):
            x_batch = (
                x_train[i : i + batch_size]
                if i + batch_size <= num_examples
                else x_train[i:]
            )
            y_batch = (
                y_train[i : i + batch_size]
                if i + batch_size <= num_examples
                else y_train[i:]
            )
            loss, acc = model.forward(x_batch, y_batch)
            model.zero_grad()
            model.backward()
            lr = lr_schedule(init_lr, iter_num, stage_num=100 / model.dp_size)
            model.update_weights(lr=lr)
            iter_num += 1

            if (iter_num + 1) % 10 == 0 and rank == 0:
                print(
                    f"Epoch:{epoch+1} iter_num:{i}/{num_examples}: Train Loss: {loss}, Train Acc: {acc}, lr_rate: {lr}"
                )
        if rank == 0:
            print("*" * 88)
            log_stats(model)
        # Evaluate
        if rank // model.mp_size == 0:
            eval_acc = 0

            if rank % model.mp_size == 0:
                print("\n" + "*" * 40 + "Evaluating" + "*" * 40)

            for i in range(0, x_test.shape[0], batch_size):
                x_batch = (
                    x_test[i : i + batch_size]
                    if i + batch_size <= num_examples
                    else x_test[i:]
                )
                y_batch = (
                    y_test[i : i + batch_size]
                    if i + batch_size <= num_examples
                    else y_test[i:]
                )
                _, acc = model.forward(x_batch, y_batch)
                eval_acc += acc * x_batch.shape[0]

            if rank % model.mp_size == 0:
                print(f"Test Acc: {eval_acc / x_test.shape[0]}")
                print("*" * 90)


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    comm = MPI.COMM_WORLD
    comm = Communicator(comm)
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    batch_size = 60
    init_lr = 0.01

    args = parser.parse_args()

    dp_size = args.dp_size
    mp_size = args.mp_size
    megatron_mp = args.megatron_mp

    if rank == 0:
        log_args(
            batch_size=batch_size,
            init_lr=init_lr,
            dp_size=dp_size,
            mp_size=mp_size,
            megatron_mp=megatron_mp,
        )

    assert dp_size * mp_size == nprocs

    mlp_model = MLPModel(
        comm=comm,
        dp_size=dp_size,
        mp_size=mp_size,
        megatron_mp=megatron_mp,
        feature_dim=784,
        hidden_dim=256,
        output_dim=10,
    )

    # load MNIST data

    MNIST_data = h5py.File("./data/MNISTdata.hdf5", "r")

    x_train = np.float32(MNIST_data["x_train"])
    y_train = np.int32(np.array(MNIST_data["y_train"][:, 0]))

    x_train, y_train = split_data(
        x_train=x_train,
        y_train=y_train,
        mp_size=mp_size,
        dp_size=dp_size,
        rank=rank,
    )

    x_test = np.float32(MNIST_data["x_test"][:])
    y_test = np.int32(np.array(MNIST_data["y_test"][:, 0]))
    MNIST_data.close()

    """
    Data Preprocess
    """
    np.random.seed(15442)
    idx = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    """
    Model Initialization
    """
    train_mlp(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model=mlp_model,
        num_epoch=1,
        batch_size=int(batch_size / dp_size),
        init_lr=init_lr,
    )
