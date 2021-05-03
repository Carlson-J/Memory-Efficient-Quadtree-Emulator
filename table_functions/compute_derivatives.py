import findiff


def compute_higher_order_derivatives(table, order_of_accuracy=6):
    """
    Compute the d3f_dden2_dtemp, d3f_dden_dtemp2, and d4f_dden2_dtemp derivatives given a table with the lower order
    derivatives already computed.
    :param table: (dict) table
    :param order_of_accuracy: (int) order of accuracy for derivatives to be computed (related to stencil size)
    :return: (dict) table with extra derivative info added.
    """
    # compute numerical derivatives
    df_dx = findiff.FinDiff(1, table['den'][0, :], acc=order_of_accuracy)
    df_dy = findiff.FinDiff(0, table['temp'][:, 0], acc=order_of_accuracy)
    df2_dxy = findiff.FinDiff((0, table['temp'][:, 0]), (1, table['den'][0, :]), acc=order_of_accuracy)
    table['Table_Values']['d3f_dtemp2_dden'] = df_dy(table['Table_Values']['d2f_dtemp_dden'])
    table['Table_Values']['d3f_dden2_dtemp'] = df_dx(table['Table_Values']['d2f_dtemp_dden'])
    table['Table_Values']['d4f_dtemp2_dden2'] = df2_dxy(table['Table_Values']['d2f_dtemp_dden'])
    return table
