import numpy as np


def compute_derived_values(table):
    """
    Compute the derived values of interest form the electron-positron helmholtz free energy.
    :param table:
    :return: [Pressure, Internal Energy, Entropy, dp_dt_const_rho, de_dt_const_rho, ds_dpho_const_t]
    """
    den = table['den']
    temp = table['temp']
    # compute pressure
    P = den**2*table['Table_Values']['df_dden']
    # Compute entropy
    S = -table['Table_Values']['df_dtemp']
    # compute Internal Energy
    E = table['Table_Values']['f'] + temp * S
    # compute dp_dt_const_rho
    dp_dt_const_rho = den**2*table['Table_Values']['d2f_dtemp_dden']
    # compute de_dt_const_rho
    de_dt_const_rho = -temp * table['Table_Values']['d2f_dtemp2']
    # compute ds_dpho_const_t
    ds_dpho_const_t = -table['Table_Values']['d2f_dtemp_dden']

    return [P, E, S, dp_dt_const_rho, de_dt_const_rho, ds_dpho_const_t]
