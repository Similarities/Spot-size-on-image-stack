import matplotlib.pyplot as plt
import numpy as np
import basic_image_app
from lmfit.models import GaussianModel
import math


class SourceSize:
    def __init__(self, file):
        self.file_name = file[-19:-4]
        self.image = basic_image_app.read_image(file)
        self.index_y, self.index_x_axis = self.evaluate_index_of_maximum_in_picture()
        self.x_axis_vertical, self.line_out_vertical = self.vertical_line_out()
        self.x_axis_horizontal, self.line_out_horizontal = self.horizontal_line_out()
        self.line_out_horizontal = self.zero_offset(self.line_out_horizontal)
        self.line_out_vertical = self.zero_offset(self.line_out_vertical)
        self.sigma_temp = 0
        self.amplitude_temp = 0
        self.center_temp = 0
        self.result = np.empty([1, 4])

    def evaluate_index_of_maximum_in_picture(self):
        self.index_y, self.index_x_axis = np.where(self.image >= np.amax(self.image))
        print(np.amax(self.image), 'max', self.file_name)
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
        return self.x_axis_vertical, self.line_out_vertical

    def horizontal_line_out(self):
        self.line_out_horizontal = np.reshape(self.image[self.index_y, :], np.size(self.image[self.index_y, :]))
        self.x_axis_horizontal = np.arange(0, len(self.line_out_horizontal))
        return self.x_axis_horizontal, self.line_out_horizontal

    def evaluate_horizontal_and_vertical(self):
        self.fit_gaussian(self.x_axis_horizontal, self.line_out_horizontal, 3)
        self.result[0, 0] = self.sigma_temp
        self.result[0, 2] = self.center_temp
        self.fit_gaussian(self.x_axis_vertical, self.line_out_vertical[:, 0], 2)
        self.result[0, 1] = self.sigma_temp
        self.result[0, 3] = self.center_temp
        return self.result

    def fit_gaussian(self, array_x, array_y, figure_number):
        mod = GaussianModel()
        pars = mod.guess(array_y, x=array_x)
        out = mod.fit(array_y, pars, x=array_x)
        self.sigma_temp = out.params['sigma'].value
        self.amplitude_temp = out.params['amplitude'].value
        self.center_temp = out.params['center'].value
        self.plot_fit_function(array_x, figure_number)
        return self.sigma_temp, self.amplitude_temp, self.center_temp

    def zero_offset(self, array):
        offset = np.mean(array[1:100])
        return array[:] - offset

    def plot_fit_function(self, array_x, figure_number):
        # IMPORTANT sigma  corresponds to w(0) beamwaist = half beam aperture
        yy = np.zeros([len(array_x), 1])
        for counter, value in enumerate(array_x):
            a = (self.amplitude_temp / (self.sigma_temp * ((2 * math.pi) ** 0.5)))
            b = -(array_x[counter] - self.center_temp) ** 2
            c = 2 * self.sigma_temp ** 2
            yy[counter] = (a * math.exp(b / c))

        plt.figure(figure_number)
        plt.plot(array_x, yy)


class BatchStack:
    def __init__(self, path):
        self.folder_name = path
        self.file_list = basic_image_app.get_file_list(self.folder_name)
        self.folder_result_sigma = np.empty([1, 4])
        self.label = "px"

    def evaluate_folder(self):
        for x in self.file_list:
            SinglePicture = SourceSize(self.folder_name + x)
            result = SinglePicture.evaluate_horizontal_and_vertical()
            self.folder_result_sigma = np.vstack((self.folder_result_sigma, result))

        print('############')
        print(self.folder_name)
        self.folder_result_sigma = self.folder_result_sigma[1:]
        self.pointing_vertical()
        self.pointing_horizontal()
        self.beamwaist_to_FWHM()
        return self.folder_result_sigma, len(self.file_list)

    def beamwaist_to_FWHM(self):
        FWHM = np.empty([len(self.file_list), 2])
        FWHM[:, 0] = ((self.folder_result_sigma[:, 0] ** 2 + self.folder_result_sigma[:, 1] ** 2) ** 0.5)
        # sigma in the gaussian fit is w(z) beam-waist radius : w(z) = FWHM/(2*ln2)**0.5 (see wiki gaussian beams)
        FWHM[:, 1] = ((self.folder_result_sigma[:, 0] ** 2 + self.folder_result_sigma[:, 1] ** 2) ** 0.5) * (
                    2 * 0.69) ** 0.5
        self.folder_result_sigma = np.hstack((self.folder_result_sigma, FWHM))
        return self.folder_result_sigma

    def scale_result(self, const, label):
        self.label = label
        self.folder_result_sigma[:, :] = self.folder_result_sigma[:, :] * const
        return self.folder_result_sigma, self.label

    def pointing_vertical(self):
        mean_position = np.mean(self.folder_result_sigma[:, 2])
        self.folder_result_sigma[:, 2] = self.folder_result_sigma[:, 2] - mean_position
        return self.folder_result_sigma

    def pointing_horizontal(self):
        mean_position = np.mean(self.folder_result_sigma[:, 3])
        self.folder_result_sigma[:, 3] = self.folder_result_sigma[:, 3] - mean_position
        return self.folder_result_sigma

    def prepare_header(self, description1):
        # insert header line and change index
        names = (['file', self.folder_name, description1, "unit: ", self.label, "..."])
        parameter_info = (
            ["vertical w(z)", "  horizontal w(z)", "pointing_vertical ", "pointing_horizontal", "FWHM_v", "FWHM_h"])
        return np.vstack((parameter_info, names, self.folder_result_sigma))

    def save_data(self, description1, file_name):
        result = self.prepare_header(description1)
        print('...saving:', file_name)

        plt.figure(5)
        x_axis = np.arange(0, len(self.file_list))
        plt.scatter(x_axis, self.folder_result_sigma[:, 1], label='vertical w(z) ' +file_name[-2:], color="y")
        plt.scatter(x_axis, self.folder_result_sigma[:, 0], label='horizontal w(z) '+file_name[-2:], color="c")
        plt.xlabel('shot number')
        plt.ylabel(self.label)
        plt.ylim(400,2500)
        plt.legend()
        plt.savefig(file_name + description1 + ".png", bbox_inches="tight", dpi=500)

        plt.figure(6)
        plt.scatter(x_axis, self.folder_result_sigma[:, 3], label="pointing_horizontal " + file_name[-2:])
        plt.scatter(x_axis, self.folder_result_sigma[:, 2], label="pointing_vertical " +file_name[-2:])
        plt.xlabel("shot no")
        plt.ylabel(self.label)
        plt.ylim(-600, 600)
        plt.legend()
        plt.savefig(file_name + "pointing" + ".png", bbox_inches="tight", dpi=500)

        np.savetxt(file_name + description1 + ".txt", result, delimiter=' ',
                   header='string', comments='',
                   fmt='%s')


path = "data/Pointing2/210505/resized_1s/"
my_result = BatchStack(path)
my_result.evaluate_folder()
# rescale is factor for e.g. px or magnification size
my_result.scale_result(13.5, "um")
my_result.save_data('Pointing_2', "20210505_HHG_source_He_1s")
plt.show()
