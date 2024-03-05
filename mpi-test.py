from mpi4py import MPI
from mpi_wrapper import Communicator
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_case",
    type=str,
    help="MPI names for different toy examples",
    default="",
    choices=["allreduce", "allgather", "reduce_scatter", "split"],
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    comm = Communicator(comm)

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    args = parser.parse_args()

    if args.test_case == "allreduce":
        """
        Allreduce example
        """

        r = np.random.randint(0, 100, 10)
        rr = np.empty(10, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Allreduce(r, rr, op=MPI.MIN)

        if rank == 0:
            print("Allreduce: " + str(rr))

    elif args.test_case == "allgather":
        """
        Allgather example
        """

        r = np.random.randint(0, 100, 2)
        rr = np.empty(16, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Allgather(r, rr)

        if rank == 0:
            print("Allgather: " + str(rr))

    elif args.test_case == "reduce_scatter":
        """
        Reduce_scatter example
        """

        r = np.random.randint(0, 100, 16)
        rr = np.empty(2, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Reduce_scatter(r, rr, op=MPI.MIN)

        print("Rank " + str(rank) + " After Reduce_scatter: " + str(rr))

    elif args.test_case == "split":
        """
        Split example (group-wise reduce)
        split into 4 groups based on the modulo operation:
         Group 0: (0, 4)
         Group 1: (1, 5)
         Group 2: (2, 6)
         Group 3: (3, 7)
        """

        r = np.random.randint(0, 100, 10)
        rr = np.empty(10, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        key = rank
        color = rank % 4

        group_comm = comm.Split(key=key, color=color)

        group_comm.Barrier()
        group_comm.Allreduce(r, rr, op=MPI.MIN)

        print("Rank " + str(rank) + " After split and Allreduce: " + str(rr))

    else:
        print(f"This is rank {rank}.")
