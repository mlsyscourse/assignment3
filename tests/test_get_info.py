from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import get_info


def check_info(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    rank = input_dict["rank"]

    mp_group_idx, dp_group_idx, mp_comm, dp_comm, part_in_dim, part_out_dim = get_info(
        comm=input_dict["comm"],
        rank=rank,
        mp_size=input_dict["mp_size"],
        dp_size=input_dict["dp_size"],
        is_fc1=input_dict["is_fc1"],
        is_megatron_mp=input_dict["is_megatron_mp"],
        in_dim=input_dict["in_dim"],
        out_dim=input_dict["out_dim"],
    )

    assert mp_group_idx == expect_output_dict["mp_group_idx"][rank]
    assert dp_group_idx == expect_output_dict["dp_group_idx"][rank]
    assert part_in_dim == expect_output_dict["part_in_dim"]
    assert part_out_dim == expect_output_dict["part_out_dim"]

    local_arr = input_dict["input_array"][rank]

    mp_group_reduction_arr = np.empty_like(local_arr)
    dp_group_reduction_arr = np.empty_like(local_arr)

    mp_comm.Allreduce(local_arr, mp_group_reduction_arr, op=MPI.SUM)
    dp_comm.Allreduce(local_arr, dp_group_reduction_arr, op=MPI.SUM)

    np.testing.assert_allclose(
        actual=mp_group_reduction_arr,
        desired=expect_output_dict["mp_group_array"][dp_group_idx],
    )
    np.testing.assert_allclose(
        actual=dp_group_reduction_arr,
        desired=expect_output_dict["dp_group_array"][mp_group_idx],
    )


@pytest.mark.mpi
def test_fc1_naive():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 2,
        "dp_size": 4,
        "is_fc1": True,
        "is_megatron_mp": False,
        "in_dim": 768,
        "out_dim": 256,
        "input_array": array,
    }

    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 0,
            3: 1,
            4: 0,
            5: 1,
            6: 0,
            7: 1,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 3,
            7: 3,
        },
        "part_in_dim": 768,
        "part_out_dim": 128,
        "mp_group_array": {
            0: np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]),
            1: np.array([50, 52, 54, 56, 58, 60, 62, 64, 66, 68]),
            2: np.array([90, 92, 94, 96, 98, 100, 102, 104, 106, 108]),
            3: np.array([130, 132, 134, 136, 138, 140, 142, 144, 146, 148]),
        },
        "dp_group_array": {
            0: np.array([120, 124, 128, 132, 136, 140, 144, 148, 152, 156]),
            1: np.array([160, 164, 168, 172, 176, 180, 184, 188, 192, 196]),
        },
    }

    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )


@pytest.mark.mpi
def test_fc2_naive():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 2,
        "dp_size": 4,
        "is_fc1": False,
        "is_megatron_mp": False,
        "in_dim": 256,
        "out_dim": 10,
        "input_array": array,
    }

    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 0,
            3: 1,
            4: 0,
            5: 1,
            6: 0,
            7: 1,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 3,
            7: 3,
        },
        "part_in_dim": 256,
        "part_out_dim": 5,
        "mp_group_array": {
            0: np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]),
            1: np.array([50, 52, 54, 56, 58, 60, 62, 64, 66, 68]),
            2: np.array([90, 92, 94, 96, 98, 100, 102, 104, 106, 108]),
            3: np.array([130, 132, 134, 136, 138, 140, 142, 144, 146, 148]),
        },
        "dp_group_array": {
            0: np.array([120, 124, 128, 132, 136, 140, 144, 148, 152, 156]),
            1: np.array([160, 164, 168, 172, 176, 180, 184, 188, 192, 196]),
        },
    }

    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )


@pytest.mark.mpi
def test_fc1_megatron():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 2,
        "dp_size": 4,
        "is_fc1": True,
        "is_megatron_mp": True,
        "in_dim": 768,
        "out_dim": 256,
        "input_array": array,
    }

    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 0,
            3: 1,
            4: 0,
            5: 1,
            6: 0,
            7: 1,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 3,
            7: 3,
        },
        "part_in_dim": 768,
        "part_out_dim": 128,
        "mp_group_array": {
            0: np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]),
            1: np.array([50, 52, 54, 56, 58, 60, 62, 64, 66, 68]),
            2: np.array([90, 92, 94, 96, 98, 100, 102, 104, 106, 108]),
            3: np.array([130, 132, 134, 136, 138, 140, 142, 144, 146, 148]),
        },
        "dp_group_array": {
            0: np.array([120, 124, 128, 132, 136, 140, 144, 148, 152, 156]),
            1: np.array([160, 164, 168, 172, 176, 180, 184, 188, 192, 196]),
        },
    }

    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )


@pytest.mark.mpi
def test_fc2_megatron():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 2,
        "dp_size": 4,
        "is_fc1": False,
        "is_megatron_mp": True,
        "in_dim": 256,
        "out_dim": 10,
        "input_array": array,
    }

    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 0,
            3: 1,
            4: 0,
            5: 1,
            6: 0,
            7: 1,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 3,
            7: 3,
        },
        "part_in_dim": 128,
        "part_out_dim": 10,
        "mp_group_array": {
            0: np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]),
            1: np.array([50, 52, 54, 56, 58, 60, 62, 64, 66, 68]),
            2: np.array([90, 92, 94, 96, 98, 100, 102, 104, 106, 108]),
            3: np.array([130, 132, 134, 136, 138, 140, 142, 144, 146, 148]),
        },
        "dp_group_array": {
            0: np.array([120, 124, 128, 132, 136, 140, 144, 148, 152, 156]),
            1: np.array([160, 164, 168, 172, 176, 180, 184, 188, 192, 196]),
        },
    }

    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )
