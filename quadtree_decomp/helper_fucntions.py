import numpy as np


def find_int_type(size):
    """
    Return the numpy data type for the smallest unsigned int needed for the input size
    :param size:
    :return:
    """
    UNSIGNED_INT_8BIT_SIZE = 255
    UNSIGNED_INT_16BIT_SIZE = 65535
    UNSIGNED_INT_32BIT_SIZE = 4294967295
    if size < UNSIGNED_INT_8BIT_SIZE:
        dtype = np.uint8
    elif size < UNSIGNED_INT_16BIT_SIZE:
        dtype = np.uint16
    elif size < UNSIGNED_INT_32BIT_SIZE:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype


def encode_mapping(mapping):
    """
    Encodes a mapping using run-length encoding.
    :param mapping:
    :return: (2d array) encoded mapping
    """
    assert(np.ndim(mapping) == 1)
    # Allocate encoding array to max size
    N = len(mapping)
    encoding = np.zeros([2, N], dtype=int)
    # setup index
    i = 0
    current_var = mapping[0]
    # do a cum sum of the index for repeat values
    for j in range(N):
        if current_var == mapping[j]:
            continue
        else:
            # save mapping var and index.
            encoding[0, i] = j - 1
            encoding[1, i] = current_var
            i += 1
            current_var = mapping[j]
    # do last one
    encoding[0, i] = j
    encoding[1, i] = current_var
    i += 1

    # trim extra zeros off
    encoding = encoding[:, :i]

    return encoding


def find_mapping_value(encoding, index):
    """
    Given an index, return the mapping index
    :param encoding: a mapping of index range to mapping value
    :param index: the index to retrieve
    :return:
    """
    return np.searchsorted(encoding, index, side='left')

