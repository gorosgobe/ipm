import json
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    achieved_plot = []
    special_distance = float(sys.argv[1])
    special_counts = []
    distances = np.arange(0.005, 0.45, 0.001)
    distance_values = []
    mean_minimum_distances = []
    variance_minimum_distances = []

    for test_idx in range(2, len(sys.argv)):
        test_name = sys.argv[test_idx]
        print("Test name:", test_name)

        with open(test_name, "r") as f:
            content = json.loads(f.read())
            minimum_distances = content["min_distances"]

        mean_minimum_distance = 0
        for i_str in minimum_distances:
            dist = minimum_distances[i_str]
            distance_values.append(dist)
            mean_minimum_distance += dist
        res = np.var(np.array(distance_values))
        variance_minimum_distances.append((test_name, res))
        mean_minimum_distance /= len(minimum_distances)
        mean_minimum_distances.append((test_name, mean_minimum_distance))

        test_achieved = []
        for target_distance in distances:
            achieved = 0
            special_distance_count = 0
            for i_str in minimum_distances:
                dist = minimum_distances[i_str]
                if dist <= target_distance:
                    achieved += 1
                if dist <= special_distance:
                    special_distance_count += 1

            print("Achieved: {} -> {}/{}".format(target_distance, achieved, len(minimum_distances)))
            test_achieved.append((target_distance, achieved))
        special_counts.append((test_name, special_distance_count))
        achieved_plot.append((test_name, test_achieved))

    print(f"Count for distance {special_distance}: {special_counts}")
    print(f"MMD: {sorted(mean_minimum_distances, key=lambda x: x[1])}")
    print(f"Var MD: {variance_minimum_distances}")

    for t in achieved_plot:
        name, achieved_data = t
        target_distances, num_achieved = zip(*achieved_data)
        plt.plot(target_distances, num_achieved, label=name)
    plt.legend()
    plt.show()

