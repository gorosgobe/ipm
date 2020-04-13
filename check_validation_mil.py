import os
import pprint
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.common.test_utils import add_value_to_test

if __name__ == '__main__':
    tests = sys.argv[1:]
    num_val_losses = 20
    plot = True
    disable_std = True
    validations = OrderedDict()

    for test in tests:
        info = torch.load(f"models/pretraining_test/val_losses/{test}")
        validation_list = info["validation_list"]
        test_basename = os.path.basename(test)
        add_value_to_test(validations, test_basename, validation_list)

    # get some initial validation losses to see progression and remove epoch number
    for overall_test_name in validations:
        for idx, test_validations in enumerate(validations[overall_test_name]):
            for idx_nn, nn_validations in enumerate(test_validations):
                _epochs, validation_losses = zip(*nn_validations)
                validation_losses = list(validation_losses)
                # extend, as not all networks train for the same time, and some will train only for "patience" num epochs
                validation_losses.extend([0 for _ in range(num_val_losses - len(validation_losses))])
                validations[overall_test_name][idx][idx_nn] = validation_losses[:num_val_losses]

    limited_np_validations = OrderedDict()
    for overall_test_name in validations:
        all = []
        for test_validations in validations[overall_test_name]:
            all.append(np.array(test_validations))
        limited_np_validations[overall_test_name] = np.concatenate(all, axis=0)

    means = []
    standard_deviations = []
    for overall_test_name in limited_np_validations:
        sum = np.sum(limited_np_validations[overall_test_name], axis=0)
        div = np.count_nonzero(limited_np_validations[overall_test_name], axis=0)
        m = sum / div
        means.append((overall_test_name, m))
        squared_difference = np.square(limited_np_validations[overall_test_name] - m)
        std_sum = np.sum(squared_difference, axis=0)
        std = np.sqrt(std_sum / len(limited_np_validations[overall_test_name]))
        standard_deviations.append((overall_test_name, std))

    pprint.pprint(means)

    if plot:
        for idx, test_means in enumerate(means):
            test_means_name, test_means_values = test_means
            _, test_stds_values = standard_deviations[idx]
            plt.plot(range(1, num_val_losses + 1), test_means_values, label=test_means_name)
            if not disable_std:
                plt.fill_between(range(1, num_val_losses + 1), test_means_values-test_stds_values, test_means_values+test_stds_values, alpha=0.2)
        plt.xticks(range(1, num_val_losses + 1))
        plt.yscale("log")
        plt.xlabel("# epochs")
        plt.ylabel("Validation loss")
        plt.legend()
        plt.show()
