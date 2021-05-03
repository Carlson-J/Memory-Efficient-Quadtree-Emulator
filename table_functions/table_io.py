import h5py
import os
import numpy as np


def load_table(save_location):
    """
    Loads the EOS table.
    :param save_location: (string) path to table.
    :return: (dict) A dictionary containing all of the table values:
        {'temp', 'den',
        'Table_Values':{'f', 'df_dden','d2f_dden2','df_dtemp', 'd2f_dtemp2', 'd2f_dtemp_dden'},
        'attrs'}
    """
    # check if it is a valid file
    if not os.path.isfile(save_location):
        raise ValueError(f"No file found at '{save_location}'")

    # Load hdf5 file
    table_data = {}
    with h5py.File(save_location, 'r') as file:
        # Load density and temperature
        table_data['den'] = file["/den"][...]
        table_data['temp'] = file["/temp"][...]
        # load meta data
        table_data['attrs'] = {}
        for key in file.attrs.keys():
            table_data['attrs'][key] = file.attrs[key]
        # Load table values
        table_data['Table_Values'] = {}
        for key in file['/Table_Values'].keys():
            table_data['Table_Values'][key] = file["/Table_Values"][key][...]

    # Return table value
    return table_data


def save_table(den, temp, table_values, params, already_log=False):
    """
    Saves the eos table
    :param already_log: (bool) if data has already been transformed by log10
    :param den: (2D array) mesh grid of density variables
    :param temp: (2D array) mesh grid of temperature variables
    :param table_values: (dict of 2d arrays) Each item is a 2d array of table values.
    :param params: (dict) Parameters used to construct table
    :return: (none) The file is saved
    """
    # Check if save location is valid
    check_for_valid_location(params['save_location'], params['save_name'], force=params['force_save'])

    # Save table
    with h5py.File(params['file_location'], 'w') as file:
        if params['save_log'] and not already_log:
            file.create_dataset('den', data=np.log10(den))
            file.create_dataset('temp', data=np.log10(temp))
        else:
            file.create_dataset('den', data=den)
            file.create_dataset('temp', data=temp)
        file.attrs["random_locations"] = params['random_locations']
        file.attrs["den_spacing"] = params['den_spacing']
        file.attrs["temp_spacing"] = params['temp_spacing']
        file.attrs["den_spacing_type"] = params['den_spacing_type']
        file.attrs["temp_spacing_type"] = params['temp_spacing_type']
        file.attrs["save_log"] = params['save_log']

        # Save table vars
        vars = file.create_group('Table_Values')
        for key in table_values.keys():
            vars.create_dataset(key, data=table_values[key])
        vars.attrs["Description"] = "Variables for the Helmholtz table"
    return


def check_for_valid_location(directory, filename, force=False):
    """
    Checks if a file and folder already exist at a location. If the folder does not exist it is made. If
    a file already exists a prompt to override it is given.
    :param directory: (string) Location to save file to
    :param filename: (string) name of file (including extension)
    :param force: (bool) does not prompt to override files if true
    :return:
    """
    # Check if directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_location = directory + '/' + filename
    # Check if a file will be overwritten
    if os.path.isfile(save_location) and not force:
        override = input(f"The file '{save_location}' already exists. Do you wish to override the file? (y/n)")
        valid_input = False
        while not valid_input:
            if override == "n" or override == "N":
                valid_input = True
                print("Exiting...")
                exit(1)
            elif override == "y" or override == "Y":
                valid_input = True
            else:
                print("Not a valid input.")
                override = input(f"Do you wish to override the file? (y/n)")