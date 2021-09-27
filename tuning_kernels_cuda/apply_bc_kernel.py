import argparse
import json
from collections import OrderedDict
import numpy as np
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for apply_BC_kernel kernels')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x_inc', type=int, default=32)
    parser.add_argument('--block_size_y_inc', type=int, default=32)
    parser.add_argument('--block_size_x_fact', type=int, default=32)
    parser.add_argument('--block_size_y_fact', type=int, default=32)
    parser.add_argument('--block_size_x_0', type=int, default=32)
    parser.add_argument('--block_size_y_0', type=int, default=32)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: OrderedDict):
    global flux_dn
    global flux_dn_ref
    # increment case
    print(f"Running apply_BC_kernel (inc version) [{params['inc']['block_size_x']} {params['inc']['block_size_y']}]")
    args = [ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn]
    apply_bc_kernel_inc(flux_dn_ref, inc_flux)
    result = kt.run_kernel(kernel_name["inc"], kernel_src, problem_size, args, params, compiler_options=common.cp)
    common.compare_fields(result[5], flux_dn_ref, "flux_dn")
    # factor case
    print(f"Running apply_BC_kernel (fact version) [{params['fact']['block_size_x']} {params['fact']['block_size_y']}]")
    flux_dn = common.zero(flux_dn_size, common.type_float)
    flux_dn_ref = common.zero(flux_dn_size, common.type_float)
    args = [ncol, nlay, ngpt, top_at_1, inc_flux, factor, flux_dn]
    apply_bc_kernel_fact(flux_dn_ref, inc_flux, factor)
    result = kt.run_kernel(kernel_name["fact"], kernel_src, problem_size, args, params, compiler_options=common.cp)
    common.compare_fields(result[6], flux_dn_ref, "flux_dn")
    # zero case
    print(f"Running apply_BC_kernel (zero version) [{params['0']['block_size_x']} {params['0']['block_size_y']}]")
    flux_dn = common.zero(flux_dn_size, common.type_float)
    flux_dn_ref = common.zero(flux_dn_size, common.type_float)
    args = [ncol, nlay, ngpt, top_at_1, flux_dn]
    apply_bc_kernel_0(flux_dn_ref)
    result = kt.run_kernel(kernel_name["0"], kernel_src, problem_size, args, params, compiler_options=common.cp)
    common.compare_fields(result[4], flux_dn_ref, "flux_dn")


# Tuning
def tune():
    global flux_dn
    global flux_dn_ref
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {ngpt}"]
    # increment case
    print(f"Tuning {kernel_name['inc']}")
    args = [ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn]
    apply_bc_kernel_inc(flux_dn_ref, inc_flux)
    answer = [None for _ in range(0, len(args))]
    answer[5] = flux_dn_ref
    result, env = kt.tune_kernel(kernel_name["inc"], kernel_src, problem_size, args, tune_params, answer=answer,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_apply_BC_kernel_inc.json") as fp:
        json.dump(result, fp)
    # factor case
    print(f"Tuning {kernel_name['fact']}")
    flux_dn = common.zero(flux_dn_size, common.type_float)
    flux_dn_ref = common.zero(flux_dn_size, common.type_float)
    args = [ncol, nlay, ngpt, top_at_1, inc_flux, factor, flux_dn]
    apply_bc_kernel_fact(flux_dn_ref, inc_flux, factor)
    answer = [None for _ in range(0, len(args))]
    answer[6] = flux_dn_ref
    result, env = kt.tune_kernel(kernel_name["fact"], kernel_src, problem_size, args, tune_params, answer=answer,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_apply_BC_kernel_fact.json") as fp:
        json.dump(result, fp)
    # zero case
    print(f"Tuning {kernel_name['0']}")
    flux_dn = common.zero(flux_dn_size, common.type_float)
    flux_dn_ref = common.zero(flux_dn_size, common.type_float)
    args = [ncol, nlay, ngpt, top_at_1, flux_dn]
    apply_bc_kernel_0(flux_dn_ref)
    answer = [None for _ in range(0, len(args))]
    answer[4] = flux_dn_ref
    result, env = kt.tune_kernel(kernel_name["0"], kernel_src, problem_size, args, tune_params, answer=answer,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_apply_BC_kernel_0.json") as fp:
        json.dump(result, fp)


# Python reference implementations (for correctness only)
def apply_bc_kernel_inc(output, inp):
    for icol in range(0, ncol):
        for igpt in range(0, ngpt):
            if top_at_1 == 1:
                item = (igpt * ncol * (nlay + 1)) + icol
            else:
                item = (igpt * ncol * (nlay + 1)) + (nlay * ncol) + icol
            output[item] = inp[item]


def apply_bc_kernel_fact(output, inp, fact):
    for icol in range(0, ncol):
        for igpt in range(0, ngpt):
            if top_at_1 == 1:
                item = (igpt * ncol * (nlay + 1)) + icol
            else:
                item = (igpt * ncol * (nlay + 1)) + (nlay * ncol) + icol
            output[item] = inp[item] * fact[icol]


def apply_bc_kernel_0(output):
    for icol in range(0, ncol):
        for igpt in range(0, ngpt):
            if top_at_1 == 1:
                item = (igpt * ncol * (nlay + 1)) + icol
            else:
                item = (igpt * ncol * (nlay + 1)) + (nlay * ncol) + icol
            output[item] = 0.0


if __name__ == '__main__':
    command_line = parse_command_line()

    # Load src and change names for tuning
    kernels_file = common.dir_name + '../src_kernels_cuda/rte_solver_kernels.cu'
    with open(kernels_file) as fp:
        kernel_src = fp.read()
    kernel_src = kernel_src.replace("apply_BC_kernel(", "apply_BC_kernel_inc(", 1)
    kernel_src = kernel_src.replace("apply_BC_kernel(", "apply_BC_kernel_fact(", 1)
    kernel_src = kernel_src.replace("apply_BC_kernel(", "apply_BC_kernel_0(", 1)

    # Input
    ncol = common.type_int(512)
    nlay = common.type_int(140)
    ngpt = common.type_int(224)
    top_at_1 = common.type_bool(1)
    flux_dn_size = ncol * (nlay + 1) * ngpt
    inc_flux_size = ncol * ngpt
    inc_flux = np.random.random(inc_flux_size).astype(common.type_float)
    factor = np.random.random(ncol).astype(common.type_float)
    # Output
    flux_dn = common.zero(flux_dn_size, common.type_float)
    flux_dn_ref = common.zero(flux_dn_size, common.type_float)

    kernel_name = OrderedDict()
    kernel_name["inc"] = f"apply_BC_kernel_inc<{common.str_float}>"
    kernel_name["fact"] = f"apply_BC_kernel_fact<{common.str_float}>"
    kernel_name["0"] = f"apply_BC_kernel_0<{common.str_float}>"
    problem_size = (ncol, ngpt)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = OrderedDict()
        parameters["inc"] = OrderedDict()
        parameters["fact"] = OrderedDict()
        parameters["0"] = OrderedDict()
        if command_line.best_configuration:
            best_configuration = common.best_configuration("apply_BC_kernel_inc.json")
            parameters["inc"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["inc"]['block_size_y'] = best_configuration["block_size_y"]
            best_configuration = common.best_configuration("apply_BC_kernel_fact.json")
            parameters["fact"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["fact"]['block_size_y'] = best_configuration["block_size_y"]
            best_configuration = common.best_configuration("apply_BC_kernel_0.json")
            parameters["0"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["0"]['block_size_y'] = best_configuration["block_size_y"]
        else:
            parameters["inc"]['block_size_x'] = command_line.block_size_x_inc
            parameters["inc"]['block_size_y'] = command_line.block_size_y_inc
            parameters["fact"]['block_size_x'] = command_line.block_size_x_fact
            parameters["fact"]['block_size_y'] = command_line.block_size_y_fact
            parameters["0"]['block_size_x'] = command_line.block_size_x_0
            parameters["0"]['block_size_y'] = command_line.block_size_y_0
        run_and_test(parameters)
