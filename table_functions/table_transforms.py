import numpy as np
import findiff
from .compute_derivatives import compute_higher_order_derivatives


def log_transform(table, use_abs=False, return_sign=False, only_domain=False, order_of_accuracy=6):
    """
    Transform the domain, Y and X, and f and its derivatives into log space. All f values must be positive if use_abs
    is False.
    Note: for the derivatives that are higher than 2nd order, numerical values are computed using the second order
    cross derivative.
    :param table: (dict)
    :param use_abs: (bool) take the absolute value before transform. The derivatives are adjusted accordingly.
    :param return_sign: (bool) return the original sign of f
    :param only_domain: (bool) only do the log transform on the domain
    :param order_of_accuracy: (int) the order of accuracy for the numerical computation of the derivatives
    :return: (dict) log of table and (2d array) original sign values
    """
    # Save original sign of f
    if use_abs:
        sign = np.sign(table['Table_Values']['f'][...])
    else:
        sign = np.ones_like(table['Table_Values']['f'][...])

    # Convert derivatives into log space
    # # setup shallow copies for easy coding/interoperation
    t = table['temp'][...].copy()
    d = table['den'][...].copy()
    if only_domain:
        assert (not use_abs)
        table["Table_Values"]['d2f_dtemp2'][...] *= 1.0 / (np.log(10) * t) ** 2
        table["Table_Values"]['d2f_dden2'][...] *= 1.0 / (np.log(10) * d) ** 2
        table["Table_Values"]['d2f_dtemp_dden'][...] *= 1.0 / (np.log(10) * t) * 1.0 / (np.log(10) * d)
        table["Table_Values"]['df_dtemp'][...] *= 1.0 / (np.log(10) * t)
        table["Table_Values"]['df_dden'][...] *= 1.0 / (np.log(10) * d)
        table['temp'][...] = np.log10(table['temp'][...])
        table['den'][...] = np.log10(table['den'][...])
    else:
        # Make sure that all values of Y are positive
        f = table['Table_Values']['f'][...].copy() * sign
        assert (np.min(f) > 0)
        dfdt = table["Table_Values"]['df_dtemp'][...].copy() * sign
        dfdd = table["Table_Values"]['df_dden'][...].copy() * sign
        d2fdt2 = table["Table_Values"]['d2f_dtemp2'][...].copy() * sign
        d2fdtd = table["Table_Values"]['d2f_dtemp_dden'][...].copy() * sign
        d2fdd2 = table["Table_Values"]['d2f_dden2'][...].copy() * sign
        table["Table_Values"]['d2f_dtemp2'][...] = np.log(10) * t * (f * (dfdt + d2fdt2 * t) - dfdt ** 2 * t) / f ** 2
        table["Table_Values"]['d2f_dden2'][...] = np.log(10) * d * (f * (dfdd + d2fdd2 * d) - dfdd ** 2 * d) / f ** 2
        table["Table_Values"]['d2f_dtemp_dden'][...] = np.log(10) * d * t * (
                    f * d2fdtd - dfdt * dfdd) / f ** 2
        table["Table_Values"]['df_dtemp'][...] = dfdt * t / f
        table["Table_Values"]['df_dden'][...] = dfdd * d / f
        table['Table_Values']['f'][...] = np.log10(f)
        table['temp'][...] = np.log10(table['temp'][...])
        table['den'][...] = np.log10(table['den'][...])

        # compute numerical derivatives
        table = compute_higher_order_derivatives(table, order_of_accuracy=order_of_accuracy)

    if return_sign:
        return table, sign
    else:
        return table
