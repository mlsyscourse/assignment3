def log_args(*args, **kwargs):
    print("*" * 40 + "Training Args" + "*" * 40)
    for k, v in kwargs.items():
        print(f"{k:<12}" + "-" * 60 + f"{v:>8}")
    print("*" * 93 + "\n" * 2)


def log_stats(model):
    print(
        f"\n\nEstimated Communication Breakdown for MP-{model.mp_size}, DP-{model.dp_size}:"
    )
    print("=" * 30)
    print(
        f"fc1: DP={model.fc1.dp_comm.total_bytes_transferred/2**20:.2f}MB, MP={model.fc1.mp_comm.total_bytes_transferred/2**20:.2f}MB"
    )
    print("-" * 30)
    print(
        f"fc2: DP={model.fc2.dp_comm.total_bytes_transferred/2**20:.2f}MB, MP={model.fc2.mp_comm.total_bytes_transferred/2**20:.2f}MB"
    )
    print("=" * 30)
    print(
        f"\n\nEstimated Peak Memory Breakdown for MP-{model.mp_size}, DP-{model.dp_size}:"
    )
    print("=" * 50)
    print(
        f"fc1: Forward pass={model.fc1.f_peak_memory_usage.peak_memory/2**20:.2f}MB, Backward pass={model.fc1.b_peak_memory_usage.peak_memory/2**20:.2f}MB"
    )
    print("-" * 50)
    print(
        f"fc2: Forward pass={model.fc2.f_peak_memory_usage.peak_memory/2**20:.2f}MB, Backward pass={model.fc2.b_peak_memory_usage.peak_memory/2**20:.2f}MB"
    )
    print("=" * 50)
