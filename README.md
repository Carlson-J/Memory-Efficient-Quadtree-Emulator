# Memory-Efficient-Quadtree-Emulator
Code to accompany the Memory-efficient emulation of physical tabular data using quadtree decomposition 2021 paper.

## Needed python modules
```
os
sys
numpy
matplotlib
copy
h5py
numba
random
findiff
```

## Creating/formating the interpolation data
The quadtree emulator expects the input data to be in a python dictionary with the following structure
```
table = {
'den': (2d-array) x values in linear space
  'temp': (2d-array) y values in linear space
  'Table_Values': {
    'f': (2d-array) function values
    'df_dtemp': (2d-array)
    'df_dden': (2d-array)
    'd2f_dtemp_dden': (2d-array)
    'd2f_dtemp2': (2d-array)
    'd2f_dden2': (2d-array)
    'd3f_dtemp2_dden': (2d-array)
    'd3f_dden2_dtemp': (2d-array)
    'd4f_dtemp2_dden2': (2d-array)
    }
}
```
Functions for transforming, masking, saving, and loading tables are located in the table_functions directory. 

As a proxy problem, we use the electron-positron Helmholtz free energy. We use the exact EOS function [here](https://github.com/jschwab/python-helmholtz), which gives python bindings to the fortran code written by [Frank Timmes](http://cococubed.asu.edu/code_pages/eos.shtml). From this EOS we compute the electron-positron Helmholtz free energy and its many derivatives and save them in an hdf5 file with a directory format, heretofore refered to as a table. This format is assumed through the code. The compute the free energy from the EOS we use the following 
```
table_values = helmholtz.eosfxt(dens=den, temp=temp, abar=1.0, zbar=1.0)
f = ((table_values.eele + table_values.epos) - table_values.temp * (table_values.spos + table_values.sele))
# Save derivative information. See http://cococubed.asu.edu/code_pages/eos.shtml
# Save the derivative of the helmholtz free energy in terms of temp and density
df_dden = (table_values.ppos + table_values.pele) / table_values.den ** 2
d2f_dden2 = -2 * df_dden / table_values.den + table_values.dpepd / table_values.den**2
df_dtemp = -(table_values.spos + table_values.sele)
d2f_dtemp2 = -table_values.dsept
d2f_dtemp_dden = -table_values.dsepd  # You could also use: table_values.dpt / table_values.den**2
```
Where the notation is `d{function}_d{what it is in terms of}#_d{what it is in terms of}#`, where `#` is the order of derivative for that term. If zero, it is obmitted.


## Example Code
Included in the root repo is an example of how to create an emulator for the sections of the electron-positron Helmholtz free energy as discussed in the paper. 
The training and test datasets used both in the paper and in this example are hosted on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4739173.svg)](https://doi.org/10.5281/zenodo.4739173)
. 
To run the example code, modify the following lines to have the correct location of the test and training datasets. You can also very the depth, targer accuracy, and number of points used in the estimation of the error.
Note that for the referenced datasets a maximum depth of 10 can be used.
```
if __name__ == "__main__":
    # setup table location
    training_data_location = "Put path to training hdf5 table here"
    test_data_location = "Put path to testing hdf5 table here"
    # setup inputs for emulator
    depth = 7
    accuracy = 10**-3
    num_error_estimation_points = 100
```

Currently only two interpolation schemes are fully implemented in the code, 'bi-quintic_enhanced' and 'bi-quintic_enhanced_logSpace', refering to the linear-space and log-space model classes presented in the paper. 
To change which to use edit the models list when the quadtree is setup on lines
```
# setup quadtree
err_bound = accuracy
masked_table = mask(training_data)
models = ['bi-quintic_enhanced', 'bi-quintic_enhanced_logSpace']
grid = build_quadtree(err_bound, depth, masked_table, models, normalize_error=True,
                      err_bounds=[-15, 0], estimate_error=num_error_estimation_points)
```

Running this example will produce 4 quadtree emulators, one for each section of the domain. 
These four emulators for the default settings are included in the root directory of the repo for reference.
Each section's fit and the corresponding error will also be plotted and displayed. 
The error over the entire domain is also saved, which is computed from the testing data and four emulators. For the default settings the results are as follows
![alt text](https://github.com/Carlson-J/Memory-Efficient-Quadtree-Emulator/blob/main/example_error_plot.png?raw=true)
