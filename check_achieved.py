import os
import sys
from collections import OrderedDict

import numpy as np

from lib.test_utils import load_test, add_value_to_test, get_achieved_and_target, get_latex, plot_achieved

if __name__ == '__main__':
    achieved_plot = []
    special_distance = float(sys.argv[1])
    special_counts = []
    distances = np.arange(0.005, 0.45, 0.001)

    mean_minimum_distances = OrderedDict()
    std_minimum_distances = OrderedDict()

    mean_fixed_step_distances = OrderedDict()
    std_fixed_step_distances = OrderedDict()

    for test_idx in range(2, len(sys.argv)):
        test_name = f"test_results/{sys.argv[test_idx]}"
        test_base_name = os.path.basename(test_name)
        print("Test name:", test_base_name, test_name)

        distance_values = []
        fixed_step_distance_values = []

        minimum_distances, fixed_step_distances = load_test(test_name)

        for i_str in minimum_distances:
            distance_values.append(minimum_distances[i_str])

        for i_str in fixed_step_distances:
            fixed_step_distance_values.append(fixed_step_distances[i_str])

        res_mmd_std = np.std(np.array(distance_values))
        add_value_to_test(std_minimum_distances, test_base_name, res_mmd_std)
        res_mmd_mean = np.mean(np.array(distance_values))
        add_value_to_test(mean_minimum_distances, test_base_name, res_mmd_mean)

        res_fsd_std = np.std(np.array(fixed_step_distance_values))
        add_value_to_test(std_fixed_step_distances, test_base_name, res_fsd_std)
        res_fsd_mean = np.mean(np.array(fixed_step_distance_values))
        add_value_to_test(mean_fixed_step_distances, test_base_name, res_fsd_mean)

        test_achieved, special_distance_count \
            = get_achieved_and_target(distances, minimum_distances, special_distance)

        special_counts.append((test_name, special_distance_count))
        achieved_plot.append((test_name, test_achieved))

    print(f"Count for distance {special_distance}: {special_counts}")
    print(f"MMD: {mean_minimum_distances}")
    print(f"STD MD: {std_minimum_distances}")
    print(f"MFSD: {mean_fixed_step_distances}")
    print(f"STD FSD: {std_fixed_step_distances}")

    # LaTeX
    res_mmd_latex = get_latex(mean_minimum_distances, std_minimum_distances)
    res_fsd_latex = get_latex(mean_fixed_step_distances, std_fixed_step_distances, display_all_values=False)

    print("Latex MMD:", res_mmd_latex)
    print("Latex FSD:", res_fsd_latex)
    plot_achieved(achieved_plot)
