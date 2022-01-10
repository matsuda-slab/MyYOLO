import torch
import numpy as np

def count(array, param):
    """
    param : torch tensor
    array : 要素10の配列

    param のパラメータの値を見て, その階級別に, 階級値をarrayに入れる
    (0~1 のパラメータの数が array[1] に入り, 1~2 のパラメータの数が array[2]
    に入り...)
    array[0] には, パラメータ数の合計が入る
    """

    array[0] = array[0] + param.numel()

    device = param.device 
    param_abs = abs(param)

    param_0_1 = torch.where(param_abs < 1, 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_0_1_count = torch.count_nonzero(param_0_1).item()
    array[1] = array[1] + param_0_1_count

    param_1_2 = torch.where((1 <= param_abs) & (param_abs < 2), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_1_2_count = torch.count_nonzero(param_1_2).item()
    array[2] = array[2] + param_1_2_count

    param_2_4 = torch.where((2 <= param_abs) & (param_abs < 4), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_2_4_count = torch.count_nonzero(param_2_4).item()
    array[3] = array[3] + param_2_4_count

    param_4_8 = torch.where((4 <= param_abs) & (param_abs < 8), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_4_8_count = torch.count_nonzero(param_4_8).item()
    array[4] = array[4] + param_4_8_count

    param_8_16 = torch.where((8 <= param_abs) & (param_abs < 16), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_8_16_count = torch.count_nonzero(param_8_16).item()
    array[5] = array[5] + param_8_16_count

    param_16_32 = torch.where((16 <= param_abs) & (param_abs < 32), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_16_32_count = torch.count_nonzero(param_16_32).item()
    array[6] = array[6] + param_16_32_count

    param_32_64 = torch.where((32 <= param_abs) & (param_abs < 64), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_32_64_count = torch.count_nonzero(param_32_64).item()
    array[7] = array[7] + param_32_64_count

    param_64_128 = torch.where((64 <= param_abs) & (param_abs < 128), 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_64_128_count = torch.count_nonzero(param_64_128).item()
    array[8] = array[8] + param_64_128_count

    param_128 = torch.where(128 <= param_abs, 
            torch.tensor(1, dtype=torch.float).to(device), 
            torch.tensor(0, dtype=torch.float).to(device))
    param_128_count = torch.count_nonzero(param_128).item()
    array[9] = array[9] + param_128_count

def count_np(array, param):
    """
    param : ndarray
    array : 要素10の配列

    numpy version
    """

    array[0] = array[0] + param.size

    param_abs = np.abs(param)

    param_0_1 = np.where(param_abs < 1, 1, 0)
    param_0_1_count = np.count_nonzero(param_0_1)
    array[1] = array[1] + param_0_1_count

    param_1_2 = np.where((1 <= param_abs) & (param_abs < 2), 1, 0)
    param_1_2_count = np.count_nonzero(param_1_2)
    array[2] = array[2] + param_1_2_count

    param_2_4 = np.where((2 <= param_abs) & (param_abs < 4), 1, 0)
    param_2_4_count = np.count_nonzero(param_2_4)
    array[3] = array[3] + param_2_4_count

    param_4_8 = np.where((4 <= param_abs) & (param_abs < 8), 1, 0)
    param_4_8_count = np.count_nonzero(param_4_8)
    array[4] = array[4] + param_4_8_count

    param_8_16 = np.where((8 <= param_abs) & (param_abs < 16), 1, 0)
    param_8_16_count = np.count_nonzero(param_8_16)
    array[5] = array[5] + param_8_16_count

    param_16_32 = np.where((16 <= param_abs) & (param_abs < 32), 1, 0)
    param_16_32_count = np.count_nonzero(param_16_32)
    array[6] = array[6] + param_16_32_count

    param_32_64 = np.where((32 <= param_abs) & (param_abs < 64), 1, 0)
    param_32_64_count = np.count_nonzero(param_32_64)
    array[7] = array[7] + param_32_64_count

    param_64_128 = np.where((64 <= param_abs) & (param_abs < 128), 1, 0)
    param_64_128_count = np.count_nonzero(param_64_128)
    array[8] = array[8] + param_64_128_count

    param_128 = np.where(128 <= param_abs, 1, 0)
    param_128_count = np.count_nonzero(param_128)
    array[9] = array[9] + param_128_count
