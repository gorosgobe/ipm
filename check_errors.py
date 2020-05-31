import json
import sys
import matplotlib.pyplot as plt
import argparse

import numpy as np


def get_mean_error_per_step(error_values, num_steps=29):
    # list of [cumulative_error, num trajectories with this step] per step across all trajectories
    result = [[0, 0] for _ in range(num_steps)]
    for i_str in error_values:
        # error for demonstration i_str
        demonstration_errors = error_values[i_str]
        combined_errors = demonstration_errors["combined_errors"]
        for idx, c in enumerate(combined_errors):
            if idx >= num_steps:
                # not interested in next steps, go to next demonstration
                break

            combined_error_norm = c["error_norm"]
            result[idx][0] += combined_error_norm
            result[idx][1] += 1

    error_per_trajectory = [x for (x, y) in result]
    return [x / y for (x, y) in result if y > 0], error_per_trajectory


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training")
    parser.add_argument("--tests", nargs="+", default=[])
    result = parser.parse_args()

    test_names = result.tests
    print("Test names:", test_names)
    test_names = list(map(lambda t: f"{t}{result.training}", test_names))
    errors_plot = []
    steps = 29  # DEPENDS ON SCENE!
    error_for_each_trajectory = []
    for t_name in test_names:
        with open(t_name, "r") as f:
            content = json.loads(f.read())
            errors = content["errors"]

        mean_error_per_step, error_per_trajectory = get_mean_error_per_step(errors, num_steps=steps)

        errors_mean = np.mean(np.array(error_per_trajectory)).item()
        errors_std = np.std(np.array(error_per_trajectory)).item()
        error_for_each_trajectory.append((t_name, errors_mean, errors_std))

        rang = range(1, len(mean_error_per_step) + 1)
        plt.plot(rang, mean_error_per_step, label=t_name)
        plt.xticks(rang)

    res_latex = ""
    count = 0
    for p in error_for_each_trajectory:
        if count % 6 == 0:
            res_latex += "\n"
        count += 1
        name, m, std = p
        res_latex += f"{round(m, 2)}/{round(std, 2)} & "

    print("EAT", error_for_each_trajectory)
    print("LaTeX:", res_latex)

    leg = plt.legend(fontsize=30)
    for i in leg.legendHandles:
        i.set_linewidth(4)
    plt.show()
