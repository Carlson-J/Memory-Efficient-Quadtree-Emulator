import numpy as np
import csv


def load_timmes_data(dat_file_location):
    """
    loads the table data for the electron-positron bi-quintic interpolation used in by timmes.
    (see helm_table.dat in https://github.com/jschwab/python-helmholtz)
    :param dat_file_location: (string) location of helm_table.dat file in the python_helmholtz module, including
        the file name.
    :return: (dict) {'temp', 'den', 'Table_Values':{ 'f', 'df_dtemp', 'df_dden', 'd2f_dtemp_dden', 'd2f_dtemp2',
        'd2f_dden2', 'd3f_dtemp2_dden', 'd3f_dden2_dtemp', 'd4f_dtemp2_dden2'}}
    """
    data = []
    LAST_LINE_OF_INTEREST = 108741  # This is where the electron-positron helmholtz free energy data stops
    # load data from text file
    with open(dat_file_location, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i >= LAST_LINE_OF_INTEREST:
                break
            data.append(list(filter(lambda a: a != "", row)))
    data = np.array(data, dtype='d')
    # Compute domain based on meta information
    Nx = 541
    Ny = 201
    X_t, Y_t = np.meshgrid(np.linspace(-12.0, 15, Nx), np.linspace(3, 13, Ny))
    X_t = 10.0 ** X_t
    Y_t = 10.0 ** Y_t
    # Format data correctly
    Z = data[:, 0].reshape([Ny, Nx])
    # convert ordering to match our scheme
    # [Zy, Zx, Zxy, Zyy, Zxx, Zyyx, Zxxy, Zyyxx]
    # f(i,j),fd(i,j),ft(i,j),fdd(i,j),ftt(i,j),fdt(i,j), fddt(i,j),fdtt(i,j),fddtt(i,j)
    indices = [2, 1, 5, 4, 3, 7, 6, 8]
    derivatives_all = [data[:, i].reshape([Ny, Nx]) for i in indices]

    # package as table
    table = {'den': X_t, 'temp': Y_t, "Table_Values": {'f': Z}}
    for i, key in enumerate(
            ['df_dtemp', 'df_dden', 'd2f_dtemp_dden', 'd2f_dtemp2', 'd2f_dden2', 'd3f_dtemp2_dden', 'd3f_dden2_dtemp',
             'd4f_dtemp2_dden2']):
        table['Table_Values'][key] = derivatives_all[i]

    return table
