from kernel_tuner.observers import BenchmarkObserver
from collections import OrderedDict
import numpy as np
import os

# Settings
type_int = np.int32
type_float = np.float64
type_bool = np.int32  # = default without `RTE_RRTMGP_USE_CBOOL`
str_float = 'float' if type_float is np.float32 else 'double'
np.set_printoptions(edgeitems=50)
# CUDA source code
dir_name = os.path.dirname(os.path.realpath(__file__)) + '/'
# CUDA compiler parameters
include = dir_name + '../include'
cp = ['-I{}'.format(include)]


# get number of registers
class RegisterObserver(BenchmarkObserver):
    def get_results(self):
        return {"num_regs": self.dev.func.num_regs}


# Compare results against reference
def compare_fields(arr1, arr2, name):
    okay = np.allclose(arr1, arr2, atol=1e-15)
    max_diff = np.abs(arr1-arr2).max()

    if okay:
        print('results for {}: OKAY!'.format(name))
    else:
        print('results for {}: NOT OKAY, max diff={}'.format(name, max_diff))


reg_observer = RegisterObserver()
metrics = OrderedDict()
metrics["registers"] = lambda p: p["num_regs"]