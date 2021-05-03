import numpy as np
from numba import jit


def construct_A(X, Y, model):
    """
    Construct the A matrix. This will be based on the model being used
    :param X: ndarray
    :param Y: ndarray
    :param model: (string)
    :return: ndarray
    """
    if 'bi-linear' == model:
        A = np.array([X, Y, np.ones_like(Y)]).transpose()
    elif 'bi-quadratic' == model:
        A = np.array([X ** 2, Y ** 2, X * Y, X, Y, np.ones_like(Y)]).transpose()
    elif 'bi-quintic' == model:
        A = np.array([X ** 5, X ** 4 * Y, X ** 3 * Y ** 2, X ** 2 * Y ** 3, X * Y ** 4, Y ** 5,
                      X ** 4, X ** 3 * Y, X ** 2 * Y ** 2, X * Y ** 3, Y ** 4,
                      X ** 3, X ** 2 * Y, X ** 1 * Y ** 2, Y ** 3,
                      X ** 2, X ** 1 * Y, Y ** 2,
                      X, Y,
                      np.ones_like(Y)
                      ]).transpose()
    else:
        raise ValueError(f"No model for '{model}' implemented!")
    return A


def create_bi_quintic_fit(X, Y, Z, derivatives):
    """
    Packs together all the values needed to do the bi_quintic fit. These are also the weights we will save.
    :param X: (2d np-array) density values
    :param Y: (2d np-array) temp values
    :param Z: (2d np-array) Helmholtz free energy values
    :param derivatives: [Zy, Zx, Zxy, Zyy, Zxx, Zyyx, Zxxy, Zyyxx]
    :return: (dict) {'temp', 'den', 'values'}
    """
    f = Z
    # unpack derivatives
    ft, fd, fdt, ftt, fdd, fdtt, fddt, fddtt = derivatives
    # get cell boundary values
    temp = np.array([Y[0, 0], Y[-1, 0]])
    den = np.array([X[0, 0], X[0, -1]])
    f_cell = np.array([[f[0, 0], f[0, -1]], [f[-1, 0], f[-1, -1]]])
    ft_cell = np.array([[ft[0, 0], ft[0, -1]], [ft[-1, 0], ft[-1, -1]]])
    fd_cell = np.array([[fd[0, 0], fd[0, -1]], [fd[-1, 0], fd[-1, -1]]])
    fdt_cell = np.array([[fdt[0, 0], fdt[0, -1]], [fdt[-1, 0], fdt[-1, -1]]])
    ftt_cell = np.array([[ftt[0, 0], ftt[0, -1]], [ftt[-1, 0], ftt[-1, -1]]])
    fdd_cell = np.array([[fdd[0, 0], fdd[0, -1]], [fdd[-1, 0], fdd[-1, -1]]])
    fdtt_cell = np.array([[fdtt[0, 0], fdtt[0, -1]], [fdtt[-1, 0], fdtt[-1, -1]]])
    fddt_cell = np.array([[fddt[0, 0], fddt[0, -1]], [fddt[-1, 0], fddt[-1, -1]]])
    fddtt_cell = np.array([[fddtt[0, 0], fddtt[0, -1]], [fddtt[-1, 0], fddtt[-1, -1]]])
    # package in a dictionary
    return {"temp": temp, "den": den, "values": np.array([f_cell, ft_cell, fd_cell, fdt_cell, ftt_cell, fdd_cell,
                                                              fdtt_cell, fddt_cell, fddtt_cell])}


def bi_quintic_wrapper(X, Y, fit, return_derivatives=False):
    """
    wraps the bi-quintic function for use in the quadtree
    :param X: (1d np-array) density values
    :param Y: (1d np-array) temp values
    :param return_derivatives: (bool)
    :return: f_out or [f_out, f_out, ft_out, fd_out, fdt_out, ftt_out, fdd_out]
    """
    # unpack vars
    f_cell, ft_cell, fd_cell, fdt_cell, ftt_cell, fdd_cell, fdtt_cell, fddt_cell, fddtt_cell = fit['values']
    temp = fit['temp']
    den = fit['den']

    # Make sure X & Y 1d
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    X = X.flatten()
    Y = Y.flatten()

    # Create output arrays
    f_out = np.empty_like(X)
    if return_derivatives:
        ft_out = np.empty_like(X)
        fd_out = np.empty_like(X)
        fdt_out = np.empty_like(X)
        ftt_out = np.empty_like(X)
        fdd_out = np.empty_like(X)

    for i in range(X.shape[0]):
        out = bi_quintic(Y[i], X[i], temp, den, f_cell, ft_cell, fd_cell, fdt_cell, ftt_cell, fdd_cell,
                         fddt_cell, fddtt_cell, fdtt_cell)
        f_out[i] = out[0]
        if return_derivatives:
            ft_out[i] = out[1]
            fd_out[i] = out[2]
            fdt_out[i] = out[3]
            ftt_out[i] = out[4]
            fdd_out[i] = out[5]
    if return_derivatives:
        return f_out, ft_out, fd_out, fdt_out, ftt_out, fdd_out
    else:
        return f_out


@jit(nopython=True)
def bi_quintic(temp, den, t, d, f, ft, fd, fdt, ftt, fdd, fddt, fddtt, fdtt):
    """
    Input the den and temp you want along with the grid of t, d, f, ft, fd, fdt, ftt, fdd that define the cell
    that den and temp are in. This is adapted from Frank Timmes's code (https://github.com/jschwab/python-helmholtz),
    but is in python instead of Fortran.
    :param temp: (float)
    :param den: (float)
    -- The following are for the cell points that den and temp are in --
    :param t: (2x1 np-array) temperature
    :param d: (2x1 np-array) density
    :param f: (2x2 np-array) helmholtz free energy
    :param ft: (2x2 np-array) derivative in terms of temp of f
    :param fd: (2x2 np-array) derivative in terms of den of f
    :param fdt: (2x2 np-array) derivative in terms of temp and den of f
    :param ftt: (2x2 np-array) derivative in terms of temp^2 of f
    :param fdd: (2x2 np-array) derivative in terms of den^2 of f
    :param fdtt: (2x2 np-array) derivative in terms of den and temp^2 of f
    :param fddtt: (2x2 np-array) derivative in terms of den^2 and temp^2 of f
    :param fddt: (2x2 np-array) derivative in terms of den^2 and temp of f
    :return: [free, df_t, df_d, df_dt, df_tt, df_dd]
    """
    ye = 1
    # bicubic hermite polynomial statement function
    i = 0
    j = 0

    def herm5(w0t, w1t, w2t, w0mt, w1mt, w2mt, w0d, w1d, w2d, w0md, w1md, w2md):
        return f[i, j] * w0d * w0t + f[i, j + 1] * w0md * w0t \
               + f[i + 1, j] * w0d * w0mt + f[i + 1, j + 1] * w0md * w0mt \
               + ft[i, j] * w0d * w1t + ft[i, j + 1] * w0md * w1t \
               + ft[i + 1, j] * w0d * w1mt + ft[i + 1, j + 1] * w0md * w1mt \
               + ftt[i, j] * w0d * w2t + ftt[i, j + 1] * w0md * w2t \
               + ftt[i + 1, j] * w0d * w2mt + ftt[i + 1, j + 1] * w0md * w2mt \
               + fd[i, j] * w1d * w0t + fd[i, j + 1] * w1md * w0t \
               + fd[i + 1, j] * w1d * w0mt + fd[i + 1, j + 1] * w1md * w0mt \
               + fdd[i, j] * w2d * w0t + fdd[i, j + 1] * w2md * w0t \
               + fdd[i + 1, j] * w2d * w0mt + fdd[i + 1, j + 1] * w2md * w0mt \
               + fdt[i, j] * w1d * w1t + fdt[i, j + 1] * w1md * w1t \
               + fdt[i + 1, j] * w1d * w1mt + fdt[i + 1, j + 1] * w1md * w1mt \
               + fddt[i, j] * w2d * w1t + fddt[i, j + 1] * w2md * w1t \
               + fddt[i + 1, j] * w2d * w1mt + fddt[i + 1, j + 1] * w2md * w1mt \
               + fdtt[i, j] * w1d * w2t + fdtt[i, j + 1] * w1md * w2t \
               + fdtt[i + 1, j] * w1d * w2mt + fdtt[i + 1, j + 1] * w1md * w2mt \
               + fddtt[i, j] * w2d * w2t + fddtt[i, j + 1] * w2md * w2t \
               + fddtt[i + 1, j] * w2d * w2mt + fddtt[i + 1, j + 1] * w2md * w2mt


    # various differences
    din = den * ye
    dt = t[j + 1] - t[j]
    dt2 = dt * dt
    dd = d[i + 1] - d[i]
    dd2 = dd * dd
    xt = max((temp - t[j]) / dt, 0.0)
    xd = max((din - d[i]) / dd, 0.0)
    mxt = 1.0 - xt
    mxd = 1.0 - xd
    dti = 1.0 / dt
    ddi = 1.0 / dd

    # evaluate the basis functions
    si0t = psi0(xt)
    si1t = psi1(xt) * dt
    si2t = psi2(xt) * dt2

    si0mt = psi0(mxt)
    si1mt = -psi1(mxt) * dt
    si2mt = psi2(mxt) * dt2

    si0d = psi0(xd)
    si1d = psi1(xd) * dd
    si2d = psi2(xd) * dd2

    si0md = psi0(mxd)
    si1md = -psi1(mxd) * dd
    si2md = psi2(mxd) * dd2

    # their first derivatives
    dsi0t = dpsi0(xt) * dti
    dsi1t = dpsi1(xt)
    dsi2t = dpsi2(xt) * dt

    dsi0mt = -dpsi0(mxt) * dti
    dsi1mt = dpsi1(mxt)
    dsi2mt = -dpsi2(mxt) * dt

    dsi0d = dpsi0(xd) * ddi
    dsi1d = dpsi1(xd)
    dsi2d = dpsi2(xd) * dd

    dsi0md = -dpsi0(mxd) * ddi
    dsi1md = dpsi1(mxd)
    dsi2md = -dpsi2(mxd) * dd
    # their second derivatives
    ddsi0t = ddpsi0(xt) / dt2
    ddsi1t = ddpsi1(xt) / dt
    ddsi2t = ddpsi2(xt)

    ddsi0mt = ddpsi0(mxt) / dt2
    ddsi1mt = -ddpsi1(mxt) / dt
    ddsi2mt = ddpsi2(mxt)

    ddsi0d = ddpsi0(xd) / dd2
    ddsi1d = ddpsi1(xd) / dd
    ddsi2d = ddpsi2(xd)

    ddsi0md = ddpsi0(mxd) / dd2
    ddsi1md = -ddpsi1(mxd) / dd
    ddsi2md = ddpsi2(mxd)

    # the free energy
    free = herm5(si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md)
    # derivative of the free energy with density
    df_d = herm5(si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md)
    # derivative of the free energy with temperature
    df_t = herm5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md)
    # second derivative free energy with to density~2
    df_dd = herm5(si0t, si1t, si2t, si0mt, si1mt, si2mt, ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md)
    # second derivative of the free energy with temperature~2
    df_tt = herm5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md)

    # second derivative of the free energy with to temperature and density
    df_dt = herm5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md)

    return free, df_t, df_d, df_dt, df_tt, df_dd


# psi0 and its derivatives
@jit(nopython=True)
def psi0(z):
    return z ** 3 * (z * (-6.0 * z + 15.0) - 10.0) + 1.0


@jit(nopython=True)
def dpsi0(z):
    return z ** 2 * (z * (-30.0 * z + 60.0) - 30.0)


@jit(nopython=True)
def ddpsi0(z):
    return z * (z * (-120.0 * z + 180.0) - 60.0)


# def psi1 and its derivatives
@jit(nopython=True)
def psi1(z):
    return z * (z ** 2 * (z * (-3.0 * z + 8.0) - 6.0) + 1.0)


@jit(nopython=True)
def dpsi1(z):
    return z * z * (z * (-15.0 * z + 32.0) - 18.0) + 1.0


@jit(nopython=True)
def ddpsi1(z):
    return z * (z * (-60.0 * z + 96.0) - 36.0)


# def psi2 and its derivatives
@jit(nopython=True)
def psi2(z):
    return 0.5 * z * z * (z * (z * (-z + 3.0) - 3.0) + 1.0)


@jit(nopython=True)
def dpsi2(z):
    return 0.5 * z * (z * (z * (-5.0 * z + 12.0) - 9.0) + 2.0)

@jit(nopython=True)
def ddpsi2(z):
    return 0.5 * (z * (z * (-20.0 * z + 36.0) - 18.0) + 2.0)
