import matplotlib.pyplot as plt
import numpy as np
import basic_image_app
from lmfit.models import GaussianModel
import math

file_data = "data/3x945ms_11k_cropped/210310_PM031616.tif"


class SourceSize:
    def __init__(self, file):
        self.file_name = file[-19:-4]
        self.image = basic_image_app.read_image(file)
        self.index_y, self.index_x_axis = self.evaluate_index_of_maximum_in_picture()
        self.x_axis_vertical, self.line_out_vertical = self.vertical_line_out()
        self.x_axis_horizontal, self.line_out_horizontal = self.horizontal_line_out()
        self.sigma_temp, self.amplitude_temp, self.center_temp = self.fit_gaussian()
        print(self.sigma_temp, self.amplitude_temp, self.center_temp)

    def evaluate_index_of_maximum_in_picture(self):
        self.index_y, self.index_x_axis = np.where(self.image >= np.amax(self.image))
        print(np.amax(self.image), 'max')

        return self.index_y, self.index_x_axis

    def plot_results(self):
        plt.figure(1)
        plt.imshow(self.image)
        plt.scatter(self.index_x_axis, self.index_y)
        plt.vlines(x=self.index_x_axis, ymin=0, ymax=len(self.image), color="y")
        plt.figure(2)
        plt.plot(self.x_axis_vertical, self.line_out_vertical, color="b")
        plt.figure(1)
        plt.hlines(y=self.index_y, xmin=0, xmax=len(self.line_out_horizontal))
        plt.figure(3)
        plt.plot(self.x_axis_horizontal, self.line_out_horizontal, color="r")

    def vertical_line_out(self):
        # seems to be horizontal  [x values for y]
        self.line_out_vertical = self.image[:, self.index_x_axis]
        self.x_axis_vertical = np.arange(0, len(self.line_out_vertical))
        # print(len(linout), np.ndim(linout))
        return self.x_axis_vertical, self.line_out_vertical

    def horizontal_line_out(self):
        self.line_out_horizontal = np.reshape(self.image[self.index_y, :], np.size(self.image[self.index_y, :]))
        self.x_axis_horizontal = np.arange(0, len(self.line_out_horizontal))
        return self.x_axis_horizontal, self.line_out_horizontal

    def fit_gaussian(self):
        mod = GaussianModel()
        pars = mod.guess(self.line_out_horizontal, x=self.x_axis_horizontal)
        out = mod.fit(self.line_out_horizontal, pars, x=self.x_axis_horizontal)
        self.sigma_temp = out.params['sigma'].value
        self.amplitude_temp = out.params['amplitude'].value
        self.center_temp = out.params['center'].value
        # print('sigma: {0} for N:{1} = {2:8.2f}nm'
        #    .format(self.sigma_temp, self.harmonic_selected, self.lambda_fundamental / self.harmonic_selected))
        self.plot_fit_function()
        return self.sigma_temp, self.amplitude_temp, self.center_temp

    def zero_offset(self, array):
        offset = np.mean(array[1:100])
        return array[:]-offset

    def plot_fit_function(self):
        self.line_out_horizontal = self.zero_offset(self.line_out_horizontal)
        # IMPORTANT sigma  corresponds to w(0) beamwaist = half beam aperture
        self.x_axis_horizontal = basic_image_app.convert_32_bit(self.x_axis_horizontal)
        yy = basic_image_app.convert_32_bit(np.zeros([len(self.x_axis_horizontal), 1]))
        for counter, value in enumerate(self.x_axis_horizontal):
            a = (self.amplitude_temp / (self.sigma_temp * ((2 * math.pi) ** 0.5)))
            b = -(self.x_axis_horizontal[counter] - self.center_temp) ** 2
            c = 2 * self.sigma_temp ** 2
            yy[counter] = (a * math.exp(b / c))

        plt.figure(3)
        plt.plot(self.x_axis_horizontal, yy)


SinglePicture = SourceSize(file_data)
SinglePicture.plot_results()
plt.show()
