import numpy as np
import h5py
from os import path
from .interpolations_schemes import bi_quintic_wrapper
from .helper_fucntions import find_mapping_value


def bi_quintic_interp(d, t, model_array, log_space):
    """
    Uses bi-quintic interpolation. Has knowledge about how the weights are stored built in.
    :param d: density (scalar)
    :param t: temp (scalar)
    :param model_array: array of values for model in cell
    :param log_space: (bool) if interp is happening in log space.
    :return:
    """
    # Unpack the weights for use in the bi-quintic interpolation scheme
    if log_space:
        model = {"den": model_array[:2], "temp": model_array[2:4], "values": model_array[4:-1].reshape([9, 2, 2]),
                 'sign': model_array[-1]}
    else:
        model = {"den": model_array[:2], "temp": model_array[2:4], "values": model_array[4:].reshape([9, 2, 2])}

    return bi_quintic_wrapper(d, t, model)


class SavedEmulator:
    def __init__(self, filename):
        """
        Loads the saved emulator and gets it ready for computing
        :param filename: (sting) location of the emulator.
        """
        assert (path.exists(filename))
        # save file size
        self.file_size_bytes = path.getsize(filename)
        # load data
        with h5py.File(filename, 'r') as file:
            # load models
            self.model_types = []
            self.models_arrays = []
            self.log_spaced = []
            self.offsets = []
            for dataset in file['models'].items():
                name = dataset[0]
                self.model_types.append(file['models'][name].attrs['model_type'].decode("utf-8"))
                self.models_arrays.append(file['models'][name][...])
                self.log_spaced.append(file['models'][name].attrs['log_space'])
                self.offsets.append(file['models'][name].attrs['offset'])
            # sort so the offsets are in order
            indices = np.argsort(self.offsets)
            self.model_types = [self.model_types[i] for i in indices]
            self.models_arrays = [self.models_arrays[i] for i in indices]
            self.log_spaced = [self.log_spaced[i] for i in indices]
            self.offsets = [self.offsets[i] for i in indices]
            # Load mapping
            self.encoding = file['mapping']['encoding'][...]
            self.mappings = file['mapping']['map_value'][...]
            # Load meta data
            self.temp_range = file['mapping']['encoding'].attrs['temp_domain'][:]
            self.den_range = file['mapping']['encoding'].attrs['den_domain'][:]
            self.depth = file['mapping']['encoding'].attrs['depth']

    def get_num_models(self):
        """
        Compute the total number of models used.
        :return: (int) number of models used in emulator
        """
        return sum([models.shape[0] for models in self.models_arrays])

    def get_memory_cost(self):
        """
        Compute memory cost from the hdf5 file size.
        :return: (int) number of bytes used by saved emulator
        """
        return self.file_size_bytes

    def __call__(self, den, temp, log_domain=False, skip_points_outside_domain=False):
        """
        Compute the function at the given temp and den
        :param den: (2d-array) densities
        :param temp: (2d-array) temperatures
        :param log_domain: (bool) take log10 of den and temp for domain search.
        :param skip_points_outside_domain: (bool) if true, skip the evaluation of points outside the emulators domain.
            A default values of 0 is returned.
        :return: (2darray) free energy
        """
        np.atleast_2d(den)
        np.atleast_2d(temp)
        # allocate solution array
        f = np.zeros_like(temp)

        # setup log transform
        def transform(x):
            if log_domain:
                return np.log10(x)
            else:
                return x

        # solve
        for j in range(den.shape[0]):
            for i in range(den.shape[1]):
                try:
                    # find model index
                    index = self.find_model(transform(den[j, i]), transform(temp[j, i]))
                    # # decode index
                    index_deconded = find_mapping_value(self.encoding, index)
                    k = self.mappings[index_deconded]
                    # Determine which type of model it is by index
                    if len(self.offsets) > 1:
                        model_list_index = next((x for x, val in enumerate(self.offsets[1:]) if val > k), len(self.offsets)-1)
                    else:
                        model_list_index = 0
                    # check to see which model it is
                    if self.model_types[model_list_index] != "bi-quintic_enhanced" and self.model_types[model_list_index] != "bi-quintic_enhanced_logSpace":
                        raise RuntimeError(f"'{self.model_types[model_list_index]}' is not currently supported.")
                    # get model
                    model_array = self.models_arrays[model_list_index][k - self.offsets[model_list_index]]
                    # check that the correct model was selected
                    self._check_domain_is_correct(transform(den[j, i]), transform(temp[j, i]),
                                                  model={"den": model_array[:2], "temp": model_array[2:4],
                                                         'log_space': self.log_spaced[model_list_index]})
                    # transform domain if needed
                    if self.log_spaced[model_list_index]:
                        d = np.log10(den[j, i])
                        t = np.log10(temp[j, i])
                    else:
                        d = den[j, i]
                        t = temp[j, i]
                    # do interpolation
                    f[j, i] = bi_quintic_interp(d, t, model_array, self.log_spaced[model_list_index])

                    # Add sign correction if needed
                    if self.log_spaced[model_list_index]:
                        f[j, i] = 10**f[j, i]*model_array[-1]

                except ValueError as err:
                    if skip_points_outside_domain:
                        continue
                    else:
                        raise err
        return f

    def _check_domain_is_correct(self, den, temp, model=None):
        """
        Check is the den and temp are within the correct range. If no model is given
        the entire domain is used.
        :param den:
        :param temp:
        :param model: (bool or model)
        :return:
        """
        EPS = 10**-14
        if model is None:
            d_range = self.den_range
            t_range = self.temp_range
        else:
            d_range = model['den']
            t_range = model['temp']
            if not model['log_space']:
                d_range = np.log10(d_range)
                t_range = np.log10(t_range)
        if ((den < d_range[0]*(1-EPS*np.sign(d_range[0])))
                or (den > d_range[1]*(1+EPS*np.sign(d_range[1])))
                or (temp < t_range[0]*(1-EPS*np.sign(t_range[0])))
                or (temp > t_range[1]*(1+EPS*np.sign(t_range[1])))):
            raise ValueError(
                f"(den, temp) : ({den},{temp}) outside of ranges den [{d_range[0]},{d_range[1]}] and temp [{t_range[0]},{t_range[1]}]")

    def find_model(self, den, temp):
        """
        Get the index for self.models to use for a given input den and temp.
        This is done by comparing the temp and then density at each level, doing a
        binary division of the index at each comparison (a binary comparison is used
        as for speed). We start with 0 and the length
        of the mapping array as the starting indices.
        # TODO: Change this to the way we do it in the paper.
        :param den: (scalar)
        :param temp: (scalar)
        :return:
        """
        # make sure it is in the given domain
        self._check_domain_is_correct(den, temp)
        # Get initial range
        N = int(4 ** self.depth)
        n = int(2**self.depth)
        j0 = int(0)
        jf = int(N)
        # Find cell in den and temp directions that point is in.
        den_cell = int(np.floor((den - self.den_range[0]) / ((self.den_range[1] - self.den_range[0]) / n)))
        temp_cell = int(np.floor((temp - self.temp_range[0]) / ((self.temp_range[1] - self.temp_range[0]) / n)))
        # limit inputs to cells inside the domain
        if den_cell >= n:
            den_cell = n-1
        if temp_cell >= n:
            temp_cell = n - 1
        if den_cell < 0:
            den_cell = 0
        if temp_cell < 0:
            temp_cell = 0
        # Update temp cell based on the fact that the origin in at max temp, not min.
        temp_cell = 2**self.depth - temp_cell - 1
        # Determine which cell in mapping array corresponds to the den_cell and temp_cell
        for i in range(1, self.depth+1):
            # sweep in the temp direction
            if temp_cell & 2 ** (self.depth - i):
                j0 = j0 + (jf - j0) // 2
            else:
                jf = jf - (jf - j0) // 2
            # sweep in the den direction
            if den_cell & 2 ** (self.depth - i):
                j0 = j0 + (jf - j0) // 2
            else:
                jf = jf - (jf - j0) // 2

        return j0

