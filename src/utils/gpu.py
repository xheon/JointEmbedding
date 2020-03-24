import os


def set_gpu(gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
