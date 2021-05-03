"""
This is an example code to illustrate how the quadtree emulator can be created,
saved, loaded, and used for interpolation. Note that in a real application the
interpolation portion would need to be written in a compiled language to be
fast.
"""
from quadtree_decomp import build_quadtree
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from table_functions import get_extent, IndexMask, load_table, Mask
from table_functions import compute_higher_order_derivatives
import numpy as np
import matplotlib.pyplot as plt
from quadtree_decomp.evaluate_saved_emulator import SavedEmulator

if __name__ == "__main__":
    # setup table location
    training_data_location = "Put path to training hdf5 table here"
    test_data_location = "Put path to testing hdf5 table here"
    # setup inputs for emulator
    depth = 7
    accuracy = 10**-3
    num_error_estimation_points = 100

    # -------------------- Training the Emulator -------------------- #
    plot_figures = True
    training_data = load_table(training_data_location)
    testing_data = load_table(test_data_location)
    # compute numerical derivatives not supplied by the EOS
    compute_higher_order_derivatives(training_data)

    # there are 4 sections of the table
    for section in range(0, 4):
        # setup plotting
        fig, axs = plt.subplots(2, 1)
        # create mask
        offset = 224
        section_size = 2**10
        offset2 = offset + section * section_size
        offset1 = offset
        N = section_size + 1
        mask = IndexMask(offset2, offset2 + N, offset1, offset1 + N)
        # setup quadtree
        err_bound = accuracy
        masked_table = mask(training_data)
        models = ['bi-quintic_enhanced', 'bi-quintic_enhanced_logSpace']
        grid = build_quadtree(err_bound, depth, masked_table, models, normalize_error=True,
                              err_bounds=[-15, 0], estimate_error=num_error_estimation_points)

        if plot_figures:
            # this will update number of models
            grid.get_leaf_nodes([])
            # setup bounds
            bounds = get_extent(mask(training_data['den']), mask(training_data['temp']))
            p1 = [[bounds[0]]]
            p2 = [[bounds[2]]]

            # Plot fit for current section
            ax1 = axs[0]
            im_function = ax1.pcolormesh(p1, p2, [[0]], shading='nearest')

            # plot interpolated values
            grid.plot_region(ax1)
            num_models = grid.get_num_models()
            ax1.set_title(f"Depth:{depth}, err:{err_bound}, Models:{grid.get_num_models()}")
            ax1.set_ylabel('y')
            # Plot errors
            ax3 = axs[1]
            im = ax3.pcolormesh(p1, p2, [[0]], shading='nearest')
            grid.plot_leaf_error(ax3)
            ax3.set_title(f"Err Plot - Depth:{depth}, err:{err_bound}, Models:{grid.get_num_models()}")

            for ax in axs.flatten():
                ax.axis('equal')

            # make color bar
            value_bounds = grid.get_bounds(log=True)
            cbar = plt.colorbar(im_function, ax=ax1)
            im_function.set_clim(vmin=value_bounds[0], vmax=value_bounds[1])
            cbar.draw_all()
            # plot error color bar
            cbar = plt.colorbar(im, ax=ax3)
            error_bounds = grid.get_error_bounds()
            im.set_clim(vmin=error_bounds[0], vmax=error_bounds[1])
            cbar.draw_all()

            plt.tight_layout()

            plt.show()

        # Save emulator for current section
        grid.save(f'./example_emulator_s{section}_err{accuracy:1.2e}.hdf5')

    # -------------------- Load emulator and test interpolation -------------------- #
    # load emulators
    models = []
    for i in range(4):
        models.append(SavedEmulator(f'./example_emulator_s{i}_err{accuracy:1.2e}.hdf5'))

    # compute memory cost of combined region in bytes
    mem_cost = np.sum([model.get_memory_cost() for model in models])

    # make model mask
    full_mask = Mask(testing_data['den'], testing_data['temp'], [models[0].den_range[0], models[-1].den_range[1]],
                     [models[0].temp_range[0], models[0].temp_range[1]], require_square=False)

    # initialize full arrays
    err_full = np.zeros_like(testing_data['den'])

    # Loop over each section
    for section in range(4):
        # create Mask
        mask = Mask(testing_data['den'], testing_data['temp'], [models[section].den_range[0],
                                                                models[section].den_range[1]],
                    [models[section].temp_range[0], models[section].temp_range[1]], require_square=False)
        # do interpolation at test points
        den = mask(testing_data['den'])
        temp = mask(testing_data['temp'])
        Z_test = models[section](den, temp, log_domain=True, skip_points_outside_domain=False)
        Z_true = mask(testing_data['Table_Values']['f'])
        # compute errors
        err_section = mask(err_full)
        err_section[:, :] = abs(Z_true - Z_test) / abs(Z_true)

    # get norm errors
    l1 = np.mean(err_full)
    l2 = np.sum(err_full ** 2 / len(err_full.flatten())) ** 0.5
    lI = np.max(err_full)

    # print out results
    print(
        f'Accuracy: {accuracy:0.1e}, mem cost: {mem_cost / 10 ** 6:0.2e} MBytes, l1: {l1:0.1e}, l2: {l2:0.1e}, lI: {lI:0.1e}')

    extent = get_extent(full_mask(testing_data['den']), full_mask(testing_data['temp']))

    # plot errors
    plt.figure()
    plt.imshow(np.log10(full_mask(err_full) + 10 ** -300), vmin=-15, vmax=0, origin='lower', extent=extent)
    plt.title("Relative error (log10)")
    plt.colorbar()
    plt.xlabel("log10(density)")
    plt.ylabel("log10(temperature)")
    plt.savefig(f"./example_error_plot.png", dpi=250)
    plt.show()
