import numpy as np
import torch
from torch import nn
import option

args=option.parse_args()


def save_best_record(test_info, file_path):
    f = open(file_path, "w")
    f.write("epoch: {}\n".format(test_info["epoch"][-1]))
    f.write(str(test_info["test_AUC"][-1]))
    f.write("\n")
    f.write(str(test_info["test_PR"][-1]))
    f.close()