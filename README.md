# Memory-Efficient-Quadtree-Emulator
Code to accompany the Memory-efficient emulation of physical tabular data using quadtree decomposition 2021 paper.

# Running the code

## Creating/formating the interpolation data
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


