import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from table_functions import Mask
from .interpolations_schemes import bi_quintic_wrapper, create_bi_quintic_fit, construct_A
import h5py
import copy
import random
import matplotlib.colors as colors
from .helper_fucntions import encode_mapping, find_int_type


class Region:
    # region_fits = []
    # region_mapping = []  # each entry is a list of leaves that use the ith model in the region_fits
    # current_region = 0
    # N = None
    # models = []
    # # linear space values
    # X = []
    # Y = []
    # Z = []
    # derivatives = []  # where the derivatives are stored for Z. This need to be set before use.
    # # log space values
    # X_l = []
    # Y_l = []
    # Z_l = []
    # derivatives_l = []  # where the derivatives are stored for Z. This need to be set before use.
    # bounds = [None, None]
    # bounds_l = [None, None]
    # err_bounds = [-15, 0]
    # root = None
    # normalize_error = False
    # fast_err_estimate = False
    # max_achieved_depth = 0
    # signs = None  # The sign of the function before the log transformation took the abs()

    def __init__(self, emulator_data, left_boundary, right_boundary, bottom_boundary, top_boundary, err_bound=0.01,
                 depth=0, max_depth=2, parent=None, id='0'):
        """
        Initialize the regions
        :param emulator_data: (EmulatorData) a reference to a data structure that contains all the data needed for
            the quadtree.
        :param left_boundary: (float)
        :param right_boundary: (float)
        :param bottom_boundary: (float)
        :param top_boundary: (float)
        :param err_bound: (float) the max error allowed. Will refine cell if error in region exceeds this
        :param depth: (int) The current level of refinement
        :param max_depth: (int) The max level of refinement
        :param parent: (Region) the region that is this regions parent
        :param id: (string) a unique id for the node. Encodes traversal mapping of tree to get to this node
        """
        # save reference to emulator data
        self.data = emulator_data
        # The parent node should reset grid values
        if parent is None:
            self.data.region_fits = []
            self.data.region_mapping = []
            self.data.current_region = 0
            self.parent = None
            self.data.root = self
        else:
            self.parent = parent
        # define boundaries
        self.rb = right_boundary
        self.lb = left_boundary
        self.tb = top_boundary
        self.bb = bottom_boundary
        # The region is the index to the model used in this region
        self.region = self.data.current_region
        self.data.current_region += 1
        self.data.region_mapping.append([])
        self.err_bound = err_bound
        self.depth = depth
        if depth > self.data.max_achieved_depth:
            self.data.max_achieved_depth = depth
        self.max_depth = max_depth
        # Initialize to leaf node
        self.leaves = None
        self.is_leaf = True
        self.id = id
        # create mask for domain
        self.mask = Mask(self.data.X, self.data.Y, [self.lb, self.rb], [self.bb, self.tb], log_transform_XY=True)
        fit, self.region_err = self._fit_region(self.mask)
        self.data.region_fits.append(fit)

        # refine until we are within err
        if self.region_err > err_bound:
            self._divide_region()

    def _divide_region(self):
        """
        Divide the region into four subregions, one for each cartesian quadrant.
        :return: [top_left, top_right, bottom_left, bottom_right]
        """
        if self.max_depth <= self.depth:
            return False
        self.is_leaf = False
        dx = (self.rb - self.lb) / 2.0
        dy = (self.tb - self.bb) / 2.0
        self.leaves = [Region(self.data, self.lb, self.lb + dx, self.tb - dy, self.tb, self.err_bound, depth=self.depth + 1,
                              max_depth=self.max_depth, parent=self, id=self.id + "0"),
                       Region(self.data, self.lb + dx, self.rb, self.tb - dy, self.tb, self.err_bound, depth=self.depth + 1,
                              max_depth=self.max_depth, parent=self, id=self.id + "1"),
                       Region(self.data, self.lb, self.lb + dx, self.bb, self.tb - dy, self.err_bound, depth=self.depth + 1,
                              max_depth=self.max_depth, parent=self, id=self.id + "2"),
                       Region(self.data, self.lb + dx, self.rb, self.bb, self.tb - dy, self.err_bound, depth=self.depth + 1,
                              max_depth=self.max_depth, parent=self, id=self.id + "3")]

    def get_region_values(self, mask, log_domain):
        """
        Get the values from the full domain that are inside the mask
        :param mask: mask function
        :param log_domain: return values in log domain.
        :return:
        """
        if log_domain:
            return mask(self.data.X_l), mask(self.data.Y_l), mask(self.data.Z_l), [mask(d) for d in self.data.derivatives_l]
        else:
            return mask(self.data.X), mask(self.data.Y), mask(self.data.Z), [mask(d) for d in self.data.derivatives]

    def _fit_region(self, mask):
        """
        Fit the current model to the region and determines if the fit error.
        :param mask: (function) maps the full domain into this region's domain.
        :return: (dict) {'theta':weight vector, 'model':(string)}
        """
        current_error = np.infty
        best_fit = None
        assert (len(self.data.models) > 0)
        for model in self.data.models:
            # check if log space model
            if len(model) > 9 and model[-9:] == "_logSpace":
                log_space = True
                model = model[:-9]
            else:
                log_space = False
            # get domain
            X, Y, Z, derivatives = self.get_region_values(mask, log_space)
            # Choose the model that has the lowest error.
            if "bi-quintic_enhanced" == model:
                # Derivatives should be layed out in self.data.derivative like so:
                # [Zy, Zx, Zxy, Zyy, Zxx, Zyyx, Zxxy, Zyyxx]
                if derivatives is None:
                    raise ValueError("Derivatives must be included when using bi-quintic_enhanced.")
                # Change X, Y, and Z to square layout.
                N = X.shape[0]
                if X.shape[1] != X.shape[0]:
                    assert (X.shape[1] != X.shape[0])
                X = X.reshape([N, N])
                Y = Y.reshape([N, N])
                Z = Z.reshape([N, N])
                # do fit
                fit = create_bi_quintic_fit(X, Y, Z, derivatives)
                fit['model'] = model
                fit['log_space'] = log_space
                # add on sign term.
                if log_space:
                    signs = mask(self.data.signs)
                    if all((signs == -1).flatten()):
                        s = -1
                    else:
                        # If it should stay positive or if there is a mix don't change the sign.
                        s = 1
                    fit['sign_term'] = s
            else:
                A = construct_A(X, Y, model)
                theta = np.linalg.lstsq(A, Z, rcond=None)[0]
                fit = {'theta': theta, 'model': model, 'log_space': log_space}

            # check if fit is best so far
            err = self._fit_lI_err(X, Y, Z, fit, log_space=log_space)
            if err < current_error:
                current_error = err
                best_fit = fit
        # Return best fit and its error.
        return best_fit, current_error

    def _update_region(self, new_region):
        """
        Update region
        :param new_region:
        :return:
        """
        self.region = new_region

    @classmethod
    def _compute_fit(cls, X, Y, fit):
        """
        Computes the predicted values in the nodes region given a fit.
        :param fit:
        :return:
        """
        if fit['model'] == "bi-quintic_enhanced":
            return bi_quintic_wrapper(X, Y, fit)
        A = construct_A(X, Y, fit['model'])
        return A @ fit['theta']

    def get_xy_array(self, mask):
        """
        Returns the 2 1d arrays for the x and y ranges. Both are in log space.
        :param mask: (mask)
        :return:
        """
        x_array = mask(self.data.X_l)[0, :]
        y_array = mask(self.data.Y_l)[:, 0]
        return x_array, y_array

    def plot_true(self, ax, log_domain=False):
        """
        Plots the correct function over the entire domain
        :param ax: (matplotlib axis)
        :return:
        """
        x_array, y_array = self.get_xy_array(self.data.root.mask)
        if log_domain:
            ax.pcolormesh(x_array, y_array, self.data.Z_l, vmin=self.data.bounds_l[0], vmax=self.data.bounds_l[1],
                          shading='nearest')
        else:
            ax.pcolormesh(x_array, y_array, self.data.Z, vmin=self.data.bounds[0], vmax=self.data.bounds[1],
                          shading='nearest')

    def plot_leaf_error(self, ax, print_norms=False):
        """
        Plots the error of all the leaf nodes.
        :param ax: (plt axis)
        :param print_norms: (bool) print the norm errors for the leaf node.
        :return:
        """
        if self.leaves is None:
            if self.data.region_fits[self.region]['log_space']:
                fit = Region._compute_fit(self.mask(self.data.X_l, flt=True), self.mask(self.data.Y_l, flt=True),
                                          self.data.region_fits[self.region]).reshape(self.mask.Ny, self.mask.Nx)
            else:
                fit = Region._compute_fit(self.mask(self.data.X, flt=True), self.mask(self.data.Y, flt=True),
                                          self.data.region_fits[self.region]).reshape(self.mask.Ny, self.mask.Nx)
            if self.data.normalize_error:
                # normalize error
                if self.data.region_fits[self.region]['log_space']:
                    # This is the relative error in linear space, which is what we care about.
                    err = np.abs(10 ** (self.mask(self.data.Z_l) - fit) - 1)
                else:
                    err = np.abs(self.mask(self.data.Z) - fit) / np.abs(self.mask(self.data.Z))
            else:
                if self.data.region_fits[self.region]['log_space']:
                    # Error in linear space
                    err = np.abs(10 ** self.mask(self.data.Z_l) - 10 ** fit)
                else:
                    err = np.abs(self.mask(self.data.Z) - fit)
            with np.errstate(divide='ignore'):
                err_log = np.log10(err)
                err_log[np.isneginf(err_log)] = -15
            if print_norms:
                print(f"L1 Norm: {np.mean(err):2.3e}, LI Norm: {np.max(err):2.3e}")
            x_array, y_array = self.get_xy_array(self.mask)
            ax.pcolormesh(x_array, y_array, err_log, vmin=self.data.err_bounds[0],
                          vmax=self.data.err_bounds[1], shading='nearest')
        self._plot_subregion(ax, error_plot=True, print_norms=print_norms)

    def plot_region(self, ax, include_domain=True, include_fit=True, plot_model_type=False, linewidth=1):
        """
        Plots a red box outlining the region. If it is a leaf node then the predicted values are plotted, else, it
        plots all leaf nodes recursively.
        :param linewidth: (float) width of lines for grid
        :param plot_model_type: (bool) if true color in the region of each leaf node with the model type used
        :param include_fit: (bool) plot the fit values
        :param include_domain: (bool) include the cell boundaries
        :param ax: (plt axes)
        :return:
        """
        if include_domain:
            ax.add_patch(
                mpl.patches.Rectangle((self.lb, self.bb), self.rb - self.lb, self.tb - self.bb, linewidth=linewidth,
                                      edgecolor='r',
                                      facecolor='none'))
        if self.leaves is None and include_fit:
            self._plot_fit(ax)
        elif self.leaves is None and plot_model_type:
            self._plot_model_type(ax)
        self._plot_subregion(ax, include_fit=include_fit, plot_model_type=plot_model_type, linewidth=linewidth, include_domain=include_domain)

    def _plot_subregion(self, ax, error_plot=False, include_fit=True, print_norms=False, plot_model_type=False,
                        linewidth=1, include_domain=True):
        """
        Plots all leaves in a node.
        :param linewidth: (float) width of lines for grid
        :param plot_model_type: (bool) if true color in the region of each leaf node with the model type used
        :param include_fit: (bool) plot the fit values
        :param error_plot: (bool) plot the error of the region
        :param ax: (plt axes)
        :param print_norms: (bool) print the norm errors for the leaf node.
        :param include_domain: (bool) The quadtree domain grid will also be plotted
        :return:
        """
        if self.leaves is not None:
            for leaf in self.leaves:
                if error_plot:
                    leaf.plot_leaf_error(ax, print_norms=print_norms)
                else:
                    leaf.plot_region(ax, include_fit=include_fit, plot_model_type=plot_model_type, linewidth=linewidth, include_domain=include_domain)

    def _plot_fit(self, ax):
        """
        Plots the fit
        :param ax:
        :return:
        """
        x_array, y_array = self.get_xy_array(self.mask)
        if self.data.region_fits[self.region]['log_space']:
            fit = Region._compute_fit(self.mask(self.data.X_l, flt=True), self.mask(self.data.Y_l, flt=True),
                                      self.data.region_fits[self.region]).reshape(self.mask.Ny, self.mask.Nx)
        else:
            fit = Region._compute_fit(self.mask(self.data.X, flt=True), self.mask(self.data.Y, flt=True),
                                      self.data.region_fits[self.region]).reshape(self.mask.Ny, self.mask.Nx)
            fit = np.log10(abs(fit))
        ax.pcolormesh(x_array, y_array, fit,
                      vmin=self.data.bounds_l[0], vmax=self.data.bounds_l[1], shading='nearest')

    def _plot_model_type(self, ax):
        """
        Plots the model types
        :param ax:
        :return:
        """
        x_array, y_array = self.data.get_xy_array(self.mask)
        model_type = Region.get_model_name(self.data.region_fits[self.region])
        i = self.data.models.index(model_type)
        Z = np.ones([len(y_array), len(x_array)]) * i
        colorsList = ['cyan', 'lightgreen']
        CustomCmap = colors.ListedColormap(colorsList)
        bounds = np.array([0, 1, 2]) - 0.5
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=2)

        return ax.pcolormesh(x_array, y_array, Z,
                             shading='nearest', cmap=CustomCmap, norm=norm)

    def _fit_lI_err(self, X, Y, Z, fit, log_space=False):
        """
        Compute the infinity norm
        :param fit:
        :return:
        """
        # flatten all arrays
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        if self.data.fast_err_estimate:
            # make sure it is 1d
            assert (np.ndim(X) == 1)
            # get random sampling of domain to estimate error with
            N = X.shape[0]
            random_indices = tuple(np.array([np.random.choice(N, self.data.fast_err_estimate)]))
            if log_space:
                # do error checking in linear space.
                err = np.abs(
                    10 ** Z[random_indices] - 10 ** Region._compute_fit(X[random_indices], Y[random_indices],
                                                                        fit).flatten())
                if self.data.normalize_error:
                    err = err / np.abs(10 ** Z[random_indices])
            else:
                err = np.abs(
                    Z[random_indices] - Region._compute_fit(X[random_indices], Y[random_indices], fit).flatten())
                if self.data.normalize_error:
                    err = err / np.abs(Z[random_indices])
        else:
            if log_space:
                err = np.abs(10 ** Z - 10 ** Region._compute_fit(X, Y, fit).flatten())
                if self.data.normalize_error:
                    err = err / np.abs(10 ** Z)
            else:
                err = np.abs(Z - Region._compute_fit(X, Y, fit).flatten())
                if self.data.normalize_error:
                    err = err / np.abs(Z)
        return np.max(err)

    def get_number_regions(self):
        """
        Get the number of regions that have one or more leafs.
        :return:
        """
        return sum([1 if len(region) > 0 else 0 for region in self.data.region_mapping])

    def get_leaf_nodes(self, leaf_nodes):
        """
        Get all the leaf nodes and add them to the list leaf_nodes
        :param leaf_nodes: (list) all of the leaf nodes will be appended to this list
        :return: (None) the leaf_nodes list will be modified
        """
        if self.leaves is not None:
            for leaf in self.leaves:
                if leaf.leaves is None:
                    leaf_nodes.append(leaf)
                    self.data.region_mapping[leaf.region].append(leaf)
                else:
                    leaf.get_leaf_nodes(leaf_nodes)
        else:
            self.data.region_mapping[self.region].append(self)

    def __str__(self):
        """
        Print a few details about the region
        :return:
        """
        return f"x:{self.lb:0.1e}-{self.rb:0.1e}, y:{self.bb:0.1e}-{self.tb:0.1e} : region {self.region}"

    def get_num_models(self):
        """
        Get the number of models, i.e., leaf nodes, in the emulator
        :return:
        """
        total = 0
        for model in self.data.region_mapping:
            if len(model) > 0:
                total += 1
        return copy.copy(total)

    def get_cell(self, x, y):
        """
        Return the cell that contains the point (x,y) (in log space)
        :param x: (float)
        :param y: (float)
        :return: (Region)
        """
        TOP_LEFT_LEAF = 0
        TOP_RIGHT_LEAF = 1
        BOTTOM_LEFT_LEAF = 2
        BOTTOM_RIGHT_LEAF = 3
        # start at root node
        current_cell = self.data.root
        while not current_cell.is_leaf:
            if x >= current_cell.lb + (current_cell.rb - current_cell.lb) / 2:
                if y >= current_cell.bb + (current_cell.tb - current_cell.bb) / 2:
                    current_cell = current_cell.leaves[TOP_RIGHT_LEAF]
                else:
                    current_cell = current_cell.leaves[BOTTOM_RIGHT_LEAF]
            else:
                if y >= current_cell.bb + (current_cell.tb - current_cell.bb) / 2:
                    current_cell = current_cell.leaves[TOP_LEFT_LEAF]
                else:
                    current_cell = current_cell.leaves[BOTTOM_LEFT_LEAF]
        return current_cell

    def get_model(self, x, y):
        """
        Finds the cell that containing the point (x,y) and returns the model
        :param x: (float) log10(den)
        :param y: (float) log10(temp)
        :return: fit
        """
        # find the cell that x,y are in.
        cell = self.get_cell(x, y)
        # get model
        return self.data.region_fits[cell.region]

    def save(self, filename):
        """
        Saves the mapping and models for later use. The mapping is saved using a compact scheme that utilizes
        run-length encoding. See quadtree paper for more details.
        :param filename: (string) where to save things.
        :return:
        """

        # Initialize mapping.
        N = 4 ** self.data.max_achieved_depth
        mapping = np.zeros(N, dtype=np.int)

        # Save models
        models = []
        for i, mapping_lst in enumerate(self.data.region_mapping):
            if len(mapping_lst) > 0:
                models.append(self.data.region_fits[i])
                model_id = len(models) - 1
                # Get domain of each leaf
                for leaf in mapping_lst:
                    j0 = 0
                    jf = N
                    for k in range(1, len(leaf.id)):
                        dj = (jf - j0) // 4
                        if leaf.id[k] == '0':
                            jf = j0 + dj
                        elif leaf.id[k] == '1':
                            j0 = j0 + dj
                            jf = j0 + dj
                        elif leaf.id[k] == '2':
                            j0 = j0 + 2 * dj
                            jf = j0 + dj
                        elif leaf.id[k] == '3':
                            j0 = j0 + 3 * dj
                            jf = j0 + dj
                        else:
                            raise ValueError(f"Invalid leaf id: {leaf.id[k]}")
                    # Set model mapping to new model location
                    mapping[j0:jf] = model_id
        n = len(models)
        # determine number of each models
        num_models = []
        for model in self.data.models:
            i = 0
            for m in models:
                model_name = m['model']
                if m['log_space']:
                    model_name += '_logSpace'
                if model_name == model:
                    i += 1
            num_models.append(i)

        # transform list into list of model arrays, which keeping indexing in mapping pointing to correct model.
        # Each model types has its own array and index. The model arrays are put together logically in index space
        # such that they are contiguous and the offsets we compute will be the start of the new model.
        offsets = [0] + list(np.cumsum(num_models)[:-1])
        model_list = [[] for m in self.data.models]
        new_mappings = np.zeros_like(mapping)
        for j, model in enumerate(models):
            # determine which model index it is and its mapping offset
            model_name = model['model']
            if model['log_space']:
                model_name += '_logSpace'
            i = self.data.models.index(model_name)
            offset = offsets[i]
            # add the model to the array
            model_list[i].append(model)
            # fix the mapping.
            new_mappings[mapping == j] = offset + len(model_list[i]) - 1

        # save new mapping
        mapping = new_mappings
        n_mapping = len(mapping)

        # convert mapping into a compressed form using a modified run length encoding
        mapping = encode_mapping(mapping)

        # Make sure all models in each array is the same
        for i, model in enumerate(self.data.models):
            if not all([Region.get_model_name(m) == self.data.models[i] for m in model_list[i]]):
                raise ValueError("Each model array must have only 1 types of model.")

        # Convert list to array
        model_array_list = []
        for j, models in enumerate(model_list):
            if len(models) > 0:
                if models[0]['model'] == "bi-quintic_enhanced":
                    if models[0]['log_space']:
                        model_array_list.append(np.zeros([len(models), len(models[0]['values']) * 4 + 4 + 1]))
                    else:
                        model_array_list.append(np.zeros([len(models), len(models[0]['values']) * 4 + 4]))
                    for i in range(len(models)):
                        model_array_list[j][i, :2] = models[i]['den'][:]
                        model_array_list[j][i, 2:4] = models[i]['temp'][:]
                        if models[0]['log_space']:
                            model_array_list[j][i, 4:-1] = models[i]['values'][:].flatten()
                            model_array_list[j][i, -1] = models[i]['sign_term']
                        else:
                            model_array_list[j][i, 4:] = models[i]['values'][:].flatten()
                else:
                    model_array_list.append(np.zeros([len(models), len(models[0]['theta'])]))
                    for i in range(len(models)):
                        model_array_list[j][i, :] = models[i]['theta'][:]

        # Save the mapping using the smallest int size needed.
        encode_dtype = find_int_type(n_mapping)
        value_dtype = find_int_type(n)
        mapping_scan_encoding = np.ndarray.astype(mapping[0, :], dtype=encode_dtype)
        mapping_value_encoding = np.ndarray.astype(mapping[1, :], dtype=value_dtype)

        # Save arrays as hdf5 files
        with h5py.File(filename, 'w') as file:
            # Save models
            model_group = file.create_group('models')
            for j, models_array in enumerate(model_array_list):
                if len(models_array) > 0:
                    dset_model = model_group.create_dataset(self.data.models[j], models_array.shape, dtype='d')
                    dset_model[...] = models_array[...]
                    dset_model.attrs['model_type'] = self.data.models[j].encode("ascii")
                    dset_model.attrs['log_space'] = model_list[j][0]['log_space']
                    if model_list[j][0]['model'] == "bi-quintic_enhanced":
                        str = "The numbering goes top left, top right, bottom left, bottom right. den0, den1, temp0, temp1, f0, f1, f2, f3, ft0, ft1, ft2, ft3, fd0, fd1, fd2, fd3, fdt0, fdt1, fdt2, fdt3, ftt0, ftt1, ftt2, ftt3, fdd0, fdd1, fdd2, fdd3, fdtt0, fdtt1, fdtt2, fdtt3, fddt0, fddt1, fddt2, fddt3, fddtt0, fddtt1, fddtt2, fddtt3"
                        if model_list[j][0]['log_space']:
                            str += ', sign'
                        dset_model.attrs['row_ordering'] = str.encode("ascii")
                    else:
                        dset_model.attrs['row_ordering'] = "Need to add"
                    dset_model.attrs['offset'] = offsets[j]
            # Save mapping
            mapping_group = file.create_group('mapping')
            dset_encoding = mapping_group.create_dataset("encoding", data=mapping_scan_encoding, dtype=encode_dtype)
            dset_value = mapping_group.create_dataset("map_value", data=mapping_value_encoding, dtype=value_dtype)
            dset_encoding.attrs['temp_domain'] = [self.data.root.bb, self.data.root.tb]
            dset_encoding.attrs['den_domain'] = [self.data.root.lb, self.data.root.rb]
            dset_encoding.attrs['depth'] = self.data.max_achieved_depth
            file.close()

    @classmethod
    def get_model_name(cls, model):
        """
        Helper function to get model name so that it matches what the input in emulator_data.models.
        :param model:
        :return:
        """
        model_name = model['model']
        if model['log_space']:
            model_name += '_logSpace'
        return model_name

    def get_bounds(self, log=False):
        """
        Get the min and max of the function over the domain
        :return:
        """
        if log:
            return self.data.bounds_l
        else:
            return self.data.bounds

    def get_error_bounds(self):
        """
        Get the min and max of the error bounds in log space
        :return:
        """
        return self.data.err_bounds

