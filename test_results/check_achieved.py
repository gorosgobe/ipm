import json
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    achieved_plot = []
    special_distance = float(sys.argv[1])
    special_counts = []
    distances = np.arange(0.005, 0.45, 0.001)

    for test_idx in range(2, len(sys.argv)):
        test_name = sys.argv[test_idx]
        print("Test name:", test_name)

        with open(test_name, "r") as f:
            content = json.loads(f.read())

        test_achieved = []
        for target_distance in distances:
            achieved = 0
            special_distance_count = 0
            for i_str in content:
                dist = content[i_str]
                if dist <= target_distance:
                    achieved += 1
                if dist <= special_distance:
                    special_distance_count += 1

            print("Achieved: {} -> {}/{}".format(target_distance, achieved, len(content)))
            test_achieved.append((target_distance, achieved))
        special_counts.append((test_name, special_distance_count))
        achieved_plot.append((test_name, test_achieved))

    print(special_counts)

    for t in achieved_plot:
        name, achieved_data = t
        target_distances, num_achieved = zip(*achieved_data)
        plt.plot(target_distances, num_achieved, label=name)
    plt.legend()
    plt.show()

