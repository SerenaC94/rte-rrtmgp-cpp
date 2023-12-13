import os

#parameters
#useful to change
num_tests = 10
difference_tolerance = 10
time_output_file = "results.txt" 

#useless to change
temp_precision_file = "precision.txt"
temp_execution_file = "times.txt"

#cannot be changed
modes = {"gpu-original":0, "cpu-serial":1, "cpu-parallel":2}
config_file = "config.txt"
num_kernel_executions = 8



#parse differences file
with open("original_differences.csv", "r") as f:
    differences_data = f.readlines()
differences = [[float(num) for num in line.split(",")] for line in differences_data[1:]]

#define a function to compare the precision of a method to the original one, checking if the relative error is below difference_tolerance
def check_precision(mode):
    with open(temp_precision_file, "r") as f:
        precision_data = f.readlines()
    precisions = [[float(line.split(" ")[6][:-1]), float(line.split(" ")[-1][:-3])] for line in precision_data]
    
    for i, precision in enumerate(precisions):
        for j, precision_type in enumerate(precision):
            original_precision_type = differences[i][j]
            if precision_type / original_precision_type > difference_tolerance:
                print(mode + ": the error is above the set tolerance!")

#define a function to parse temp_execution_file and return a list of elapsed times
def parse_execution_times():
    with open(temp_execution_file, "r") as f:
        times_data = f.readlines()
    times = [float(line.split(" ")[-1][:-3]) for line in times_data if "Time elapsed" in line]
    return times

#define a function to parse temp_execution_file and return a list of data transfer times
def parse_transfer_times():
    with open(temp_execution_file, "r") as f:
        times_data = f.readlines()
    initial_times = [float(line.split(" ")[-1][:-3]) for line in times_data if "Data transfer time" in line]
    tau_times = [float(line.split(" ")[-1][:-3]) for line in times_data if "tau transfer time" in line]
    times = [initial_times[i] + tau_times[i] for i in range(num_kernel_executions//2)]
    return times



#compile
os.system("cd ../build && make")
#link
os.system("./make_links.sh")
#initialize
os.system("python3 allsky_init.py")

#main loop
times = {}
transfer_times = [0]*(num_kernel_executions//2)
for mode in modes:
    mode_id = modes[mode]

    times[mode] = [0]*num_kernel_executions
    for _ in range(num_tests):
        #run
        os.system("echo " + str(mode_id) + " > " + config_file)
        os.system("python3 allsky_run.py > " + temp_execution_file)

        #calculate times
        new_times = parse_execution_times()
        if mode != "gpu-original":
            new_transfer_times = parse_transfer_times()
        else:
            new_transfer_times = [0]*(num_kernel_executions//2)
        for i, new_time in enumerate(new_times):
            times[mode][i] += new_time
        for i, new_time in enumerate(new_transfer_times):
            transfer_times[i] += new_time
    #average times
    for i in range(len(times[mode])):
        times[mode][i] /= num_tests

    #compare the precision
    os.system("python3 compare-to-reference.py > " + temp_precision_file)
    check_precision(mode)
for i in range(len(transfer_times)):
    transfer_times[i] /= (2*num_tests)

#calculate 
speedups = {}
speedups["gpu-over-cpu-serial"] = [times["cpu-serial"][i]/times["gpu-original"][i] for i in range(num_kernel_executions)]
speedups["gpu-over-cpu-parallel"] = [times["cpu-parallel"][i]/times["gpu-original"][i] for i in range(num_kernel_executions)]
speedups["cpu-parallel-over-serial"] = [times["cpu-serial"][i]/times["cpu-parallel"][i] for i in range(num_kernel_executions)]

#print the results
with open(time_output_file, "w") as f:
    f.write("Times (ms):\n")
    for mode in times:
        f.write(mode + ": " + str(times[mode]) + "\n")
    f.write("Speedups:\n")
    for mode in speedups:
        f.write(mode + ": " + str(speedups[mode]) + "\n")
    f.write("Transfer times (ms):\n")
    f.write(str(transfer_times) + "\n")
