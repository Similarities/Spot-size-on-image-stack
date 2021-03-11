import numpy as np
import matplotlib.pyplot as plt
import os


def load_1d_array(file, column_1, skiprows):
    data = np.loadtxt(file, skiprows=skiprows, usecols=(column_1,))
    return data


def stack_arrays(array_1, array_2, axis):
    return np.stack((array_1, array_2), axis=axis)

def constant_array_scaling(array, constant):
    return array[:] * constant


def get_file_list(path_txt):
    data_files = []
    counter = 0
    for file in os.listdir(path_txt):
        print(file)
        try:
            if file.endswith(".txt" or ".csv"):
                data_files.append(str(file))
                counter = counter + 1
            else:
                print("only other files found")
        except Exception as e:
            raise e
    return data_files

def plot_range_of_array(array_x, array_y, x_min, x_max):
    plt.plot(array_x, array_y)
    plt.xlim(x_min, x_max)






