import numpy as np
from table_functions import log_transform
from .quadtree import Region
from table_functions import get_extent
import copy


def build_quadtree(err_bound, max_depth, table, models, normalize_error=True, err_bounds=(-15, 0),
                   estimate_error=False):
    """
    This builds an emulator using a quadtree and the model classes in the model list given.
    :param err_bound: (float) this is the error bound where, if violated, the quadtree will refine, unless the max
        depth has been reached.
    :param max_depth: (int) max depth of the quadtree
    :param table: (dict) Contains the data for the table including
        'den': (2d-array) x values in linear space
        'temp': (2d-array) y values in linear space
        'Table_Values': (dict)
            'f': (2d-array) function values
            'df_dtemp': (2d-array)
            'df_dden': (2d-array)
            'd2f_dtemp_dden': (2d-array)
            'd2f_dtemp2': (2d-array)
            'd2f_dden2': (2d-array)
            'd3f_dtemp2_dden': (2d-array)
            'd3f_dden2_dtemp': (2d-array)
            'd4f_dtemp2_dden2': (2d-array)
    :param models: list(strings) : a list of models to consider when fitting. If the fit should be done in
        the log space add the string '_logSpace' to the end of the model name.
    :param normalize_error: (bool) if true, normalize the error by dividing it by |f|
    :param err_bounds: [(float), (float)]: min and max log error when plotting.
    :param estimate_error: (int) if 0 then the full domain is used for computing the error. If not 0,
        it is the number of points to consider when computing the error. The points are chosen randomly
        from all available points in the domain.
    """
    # get log domain
    table_log, signs = log_transform(copy.deepcopy(table), use_abs=True, return_sign=True)

    # unpack derivatives
    derivatives = []
    derivatives_log = []
    for i, key in enumerate(
            ['df_dtemp', 'df_dden', 'd2f_dtemp_dden', 'd2f_dtemp2', 'd2f_dden2', 'd3f_dtemp2_dden',
             'd3f_dden2_dtemp',
             'd4f_dtemp2_dden2']):
        derivatives.append(table['Table_Values'][key])
        derivatives_log.append(table_log['Table_Values'][key])

    # create data object that emulator will use
    emulator_data = EmulatorData()

    # Set linear domain
    emulator_data.X = table['den']
    emulator_data.Y = table['temp']
    emulator_data.Z = table['Table_Values']['f']
    emulator_data.derivatives = derivatives
    emulator_data.bounds = [np.min(table['Table_Values']['f']),
                     np.max(table['Table_Values']['f'])]
    # Set log domain
    emulator_data.X_l = table_log['den']
    emulator_data.Y_l = table_log['temp']
    emulator_data.Z_l = table_log['Table_Values']['f']
    emulator_data.derivatives_l = derivatives_log
    emulator_data.bounds_l = [np.min(table_log['Table_Values']['f']),
                     np.max(table_log['Table_Values']['f'])]
    emulator_data.signs = signs
    # Save model list to use
    emulator_data.models = models
    # Set other parameters
    emulator_data.normalize_error = normalize_error
    emulator_data.err_bounds = err_bounds
    emulator_data.fast_err_estimate = estimate_error
    emulator_data.bounds = [np.min(emulator_data.Z), np.max(emulator_data.Z)]
    emulator_data.bounds_l = [np.min(emulator_data.Z_l), np.max(emulator_data.Z_l)]

    bounds = get_extent(table['den'], table['temp'])
    Grid = Region(emulator_data, bounds[0], bounds[1], bounds[2], bounds[3], err_bound=err_bound, max_depth=max_depth,
                  parent=None)
    return Grid


class EmulatorData:
    def __init__(self):
        """
        This object holds the data needed for the emulator. It will be shared across all nodes in the
        quadtree.
        """
        self.region_fits = []
        self.region_mapping = []  # each entry is a list of leaves that use the ith model in the region_fits
        self.current_region = 0
        self.N = None
        self.models = []
        # linear space values
        self.X = []
        self.Y = []
        self.Z = []
        self.derivatives = []  # where the derivatives are stored for Z. This need to be set before use.
        # log space values
        self.X_l = []
        self.Y_l = []
        self.Z_l = []
        self.derivatives_l = []  # where the derivatives are stored for Z. This need to be set before use.
        self.bounds = [None, None]
        self.bounds_l = [None, None]
        self.err_bounds = [-15, 0]
        self.root = None
        self.normalize_error = False
        self.fast_err_estimate = False
        self.max_achieved_depth = 0
        self.signs = None  # The sign of the function before the log transformation took the abs()
