import matplotlib.pyplot as plt
import numpy as np
import os


def read_image(file):
    return plt.imread(file)


def get_file_list(path_picture):
    tif_files = []
    counter = 0
    for file in os.listdir(path_picture):
        #print(file)
        try:
            if file.endswith(".tif") or file.endswith('.tiff') or file.endswith('.TIF'):
                tif_files.append(str(file))
                counter = counter + 1
            else:
                print("only other files found")
        except Exception as e:
            raise e
    return tif_files


def convert_32_bit(picture):
    return np.float32(picture)


class ImageStackMeanValue:

    def __init__(self, file_list, file_path):
        self.file_list = file_list
        self.file_path = file_path
        self.result = np.zeros([512, 512])

    def average_stack(self):

        for x in self.file_list:
            x = str(self.file_path + '/' + x)
            picture_x = plt.imread(x)
            picture_x = convert_32_bit(picture_x)
            self.result = self.result + picture_x

        self.result = self.result / (len(self.file_list))
        return self.result


class SingleImageOpen:
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path

    def return_single_image(self):
        picture = read_image(str(self.file_path + '/' + self.file_name))
        return convert_32_bit(picture)



#array = read_image("data/35A_4x4_3s_2muAl/20210730_4.TIF")
#plt.imshow(array[240:400, 46:200])
#plt.show()
