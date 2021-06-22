import matplotlib.pyplot as plt
import numpy as np
import basic_image_app
from lmfit.models import GaussianModel
import math


class SourceSize:
    def __init__(self, file, width_area):
        self.file_name = file[-19:-4]
        self.image = basic_image_app.read_image(file)
        self.index_y, self.index_x_axis = self.evaluate_index_of_maximum_in_picture()
        self.width_area = width_area
        self.center_of_intensity()
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
        # sometimes np.where has more than one coordinate:
        self.index_y = self.index_y[0]
        self.index_x_axis = self.index_x_axis[0]
        print(np.amax(self.image), 'max', self.file_name, self.index_y, self.index_x_axis)
        return self.index_y, self.index_x_axis

    def center_lineout_on_image(self):
        plt.figure(12)
        plt.imshow(self.image[int(self.index_y - self.width_area / 2): int(self.index_y + self.width_area / 2),
                   int(self.index_x_axis - self.width_area / 2): int(self.index_x_axis + self.width_area / 2)])

    def center_of_intensity(self):
        print('before', self.index_y, self.index_x_axis)
        y_up = int(self.index_y - self.width_area / 2)
        y_down = int(y_up + self.width_area)
        x_left = int(self.index_x_axis - self.width_area / 2)
        x_right = int(x_left + self.width_area)
        sum_of_y_line = np.sum(self.image[y_up: y_down, x_left:x_right], axis=1)
        sum_of_x_line = np.sum(self.image[y_up: y_down, x_left:x_right], axis=0)
        plt.legend()
        self.index_y = int(np.where(sum_of_y_line[:] >= np.amax(sum_of_y_line))[0] - self.width_area / 2 + self.index_y)
        self.index_x_axis = int(
            np.where(sum_of_x_line[:] >= np.amax(sum_of_x_line))[0] - self.width_area / 2 + self.index_x_axis)
        print('after:', self.index_y, self.index_x_axis)
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
        self.fit_gaussian(self.x_axis_vertical, self.line_out_vertical, 2)
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
        offset = np.mean(array[0:1])
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
        if figure_number == 2:
            plt.plot(self.x_axis_vertical, self.line_out_vertical)
        else:
            plt.plot(self.x_axis_horizontal, self.line_out_horizontal)


class BatchStack:
    def __init__(self, path, width):
        self.folder_name = path
        self.file_list = basic_image_app.get_file_list(self.folder_name)
        self.folder_result_sigma = np.empty([1, 4])
        self.label = "px"
        self.width = width

    def evaluate_folder(self):
        for x in self.file_list:
            print("file:", x)
            SinglePicture = SourceSize(self.folder_name + x, self.width)
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
        print("vertical")
        self.statistics_pointing(self.folder_result_sigma[:, 2])
        self.folder_result_sigma[:, 2] = self.folder_result_sigma[:, 2] - mean_position

        return self.folder_result_sigma

    def pointing_horizontal(self):
        mean_position = np.mean(self.folder_result_sigma[:, 3])
        print("horizontal")
        self.statistics_pointing(self.folder_result_sigma[:, 3])
        self.folder_result_sigma[:, 3] = self.folder_result_sigma[:, 3] - mean_position
        return self.folder_result_sigma

    def statistics_pointing(self, array):
        print('pointing mean in px:', np.mean(array), 'std in px', np.std(array))

    def statistics_divergence(self, array):
        print("divergence mean in mrad:",np.mean(array), 'std', np.std(array))


    def prepare_header(self, description1):
        # insert header line and change index
        self.statistics_divergence(self.folder_result_sigma[:,0])
        self.statistics_divergence(self.folder_result_sigma[:,1])
        names = (['file', self.folder_name, description1, "unit: ", self.label, "..."])
        parameter_info = (
            ["vertical w(z)", "  horizontal w(z)", "pointing_vertical ", "pointing_horizontal", "FWHM_v", "FWHM_h"])
        return np.vstack((parameter_info, names, self.folder_result_sigma))

    def save_data(self, description1, file_name):
        result = self.prepare_header(description1)
        print('...saving:', file_name)

        plt.figure(5)
        x_axis = np.arange(0, len(self.file_list))
        plt.scatter(x_axis, self.folder_result_sigma[:, 1], label='vertical w(z) ' + file_name[-18:-5], color="y")
        plt.scatter(x_axis, self.folder_result_sigma[:, 0], label='horizontal w(z) ' + file_name[-18:-5], color="c")
        plt.xlabel('shot number')
        plt.ylabel(self.label)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(file_name + description1 + ".png", bbox_inches="tight", dpi=500)

        plt.figure(6)
        plt.scatter(x_axis, self.folder_result_sigma[:, 3], label="pointing_horizontal " + file_name[-18:-5])
        plt.scatter(x_axis, self.folder_result_sigma[:, 2], label="pointing_vertical " + file_name[-18:-5])
        plt.xlabel("shot no")
        plt.ylabel(self.label)
        plt.ylim(-0.3, 0.3)
        plt.legend()
        plt.savefig(file_name + "pointing" + ".png", bbox_inches="tight", dpi=500)

        np.savetxt(file_name + description1 + ".txt", result, delimiter=' ',
                   header='string', comments='',
                   fmt='%s')


path = "data/210611_d_roi/"
# requires, path, width_of area to calculate center of intensity (must be in row of 2)
my_result = BatchStack(path, 100)
my_result.evaluate_folder()

# rescale is factor for e.g. px or magnification size
my_result.scale_result(13.5E-6, "m")
my_result.scale_result(1000/2.625, "mrad")
my_result.save_data('Pointing_div_', "20210611_d_")
plt.show()


print(13.5E-6*1000/2.625, 'scale px to mrad')