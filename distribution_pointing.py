import matplotlib.pyplot as plt
import numpy as np
import basic_file_app
from lmfit.models import GaussianModel
import math


class PointingDistribution:
    def __init__(self, file, column, bin_decimal, unit):
        self.file_name = file
        self.name = file[8:-3]
        self.bin_decimal = bin_decimal
        self.unit = unit
        self.array_y = np.sort(basic_file_app.load_1d_array(file, column, 3))
        self.result = np.zeros([len(self.array_y), 2])
        self.sigma = float
        self.amp = float
        self.center = float
        self.count_event()
        self.fit_gaussian()

    def bin_round(self):
        # numbers are for increasing the decimal: self.array_[:]*10 , and devide after np.round /10
        self.array_y[:] = np.round(self.array_y[:], self.bin_decimal)

        return self.array_y

    def count_event(self):
        self.array_y = self.bin_round()
        uniqueValues, occurCount = np.unique(self.array_y, return_counts=True)
        self.result = np.stack((uniqueValues, occurCount), axis=1)
        print(self.result)

        plt.figure(1)
        plt.scatter(self.result[:, 0], self.result[:, 1], label=self.name)
        plt.xlabel(self.unit)
        plt.ylabel("events")
        plt.legend()

        return self.result

    def fit_gaussian(self):
        mod = GaussianModel()
        pars = mod.guess(self.result[:, 1], x=self.result[:, 0])
        out = mod.fit(self.result[:, 1], pars, x=self.result[:, 0])
        self.sigma = out.params['sigma'].value
        self.amp = out.params['amplitude'].value
        self.center = out.params['center'].value
        print('sigma:' + str(self.sigma), 'amp:' + str(self.amp), 'center' + str(self.center))
        self.plot_fit_function(1)
        return self.sigma, self.amp, self.center

    def plot_fit_function(self, figure_number):
        # IMPORTANT sigma  corresponds to w(0) beamwaist = half beam aperture
        yy = np.zeros([len(self.result[:, 1]), 1])
        for counter, value in enumerate(self.result[:, 0]):
            a = (self.amp / (self.sigma * ((2 * math.pi) ** 0.5)))
            b = -(self.result[counter, 0] - self.center) ** 2
            c = 2 * self.sigma ** 2
            yy[counter] = (a * math.exp(b / c))
        plt.figure(figure_number)

        plt.plot(self.result[:, 0], yy, label="fit - sigma:" + str(round(self.sigma, self.bin_decimal + 1)))
        # plt.xlim(-50, 50)
        plt.legend()

    def prepare_header(self, description1):
        # insert header line and change index
        names = (['file' + self.file_name, description1])
        results = (['sigma: ' + str(self.sigma) + '  amp: ' + str(self.amp) + ' center: ' + str(self.center),
                    'unit: ' + self.unit + '  bin_size:' + str(self.bin_decimal)])
        parameter_info = (
            ["pointing" + self.unit, " events"])
        return np.vstack((names, results, parameter_info, self.result))

    def save_data(self, description1, label):
        result = self.prepare_header(description1)
        print('...saving:', self.file_name)

        np.savetxt(self.name + description1 + ".txt", result, delimiter=' ',
                   header='string', comments='',
                   fmt='%s')


# todo: give file-path
file = "results/20210623_30ms_pos3XPL.txt"

# todo: PointDistribution(filepath, column_number, decimal_for_bin, unit (str))
Horizontal = PointingDistribution(file, 2, 0, "um")
Horizontal.save_data("Horizontal")
Vertical = PointingDistribution(file, 3,0, "um")
Vertical.save_data("vertical", "um")


# todo: change filename for picture!
plt.savefig('20210623_30ms_pos3' + "pointing_distribution" + ".png", bbox_inches="tight", dpi=500)

plt.show()
