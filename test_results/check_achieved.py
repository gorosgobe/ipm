import json
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    achieved_plot = []
    distances = np.arange(0.005, 0.45, 0.001)

    for test_idx in range(1, len(sys.argv)):
        test_name = sys.argv[test_idx]

        with open(test_name, "r") as f:
            content = json.loads(f.read())

        test_achieved = []
        for target_distance in distances:
            achieved = 0
            for i_str in content:
                dist = content[i_str]
                if dist <= target_distance:
                    achieved += 1
            print("Achieved: {} -> {}/{}".format(target_distance, achieved, len(content)))
            test_achieved.append((target_distance, achieved))

        achieved_plot.append((test_name, test_achieved))

    for t in achieved_plot:
        name, achieved_data = t
        target_distances, num_achieved = zip(*achieved_data)
        plt.plot(target_distances, num_achieved, label=name)
    plt.legend()
    plt.show()

