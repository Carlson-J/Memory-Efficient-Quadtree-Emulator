# Memory-Efficient-Quadtree-Emulator
Code to accompany the Memory-efficient emulation of physical tabular data using quadtree decomposition 2021 paper.

# Running the code

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
Functions for loading, saving, and manipulating tables that are in this format and saved in hdf5 files are given in the table_function folder.

As a proxy problem, we use the electron-positron Helmholtz free energy. We use the exact EOS function [here](https://github.com/jschwab/python-helmholtz), which gives python bindings to the fortran code written by [Frank Timmes](http://cococubed.asu.edu/code_pages/eos.shtml). From this EOS we compute the electron-positron Helmholtz free energy and its many derivatives and save them in an hdf5 file with a directory format, heretofore refered to as a table. This format is assumed through the code. The compute the free energy from the EOS we use the following 
```
helmholtz.eosfxt(dens=den, temp=temp, abar=1.0, zbar=1.0)
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
Functions for transforming, masking, saving, and loading tables are located in the table_functions directory. 

## Creating the memory compact emulator

## Loading and using the memory compact emulator

# Modifying the code

## Adding different interpolation schemes


