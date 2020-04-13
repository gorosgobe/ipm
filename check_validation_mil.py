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
    for overall_test_name in limited_np_validations:
        print(limited_np_validations[overall_test_name].shape)
        sum = np.sum(limited_np_validations[overall_test_name], axis=0)
        div = np.count_nonzero(limited_np_validations[overall_test_name], axis=0)
        print("Sum shape", sum.shape)
        print("Div shape", div.shape)
        means.append((overall_test_name, sum / div))

    pprint.pprint(means)

    if plot:
        for test_means in means:
            test_means_name, test_means_values = test_means
            plt.plot(range(1, num_val_losses + 1), test_means_values, label=test_means_name)
        plt.xticks(range(1, num_val_losses + 1))
        plt.yscale("log")
        plt.xlabel("# epochs")
        plt.ylabel("Validation loss")
        plt.legend()
        plt.show()
