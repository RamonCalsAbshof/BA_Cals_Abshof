import numpy as np
import pandas
import random
import os

def generate_1d_data(num_timestamps=8, num_outliers=5, save_path=None):
    random.seed(10) # pakdd
    # random.seed(8)  # mldm_long
    #
    # start_points = [0.85, 0.65, 0.4, 0.15]  # mldm_long

    start_points = [0.8, 0.65, 0.3, 0.15] # pakdd
    max_devs = [0.1, 0.2, 0.1, 0.15]
    num_members = [5, 7, 8, 6]

    data = []

    object_id = 1
    counter = 0
    for start_point, max_dev in zip(start_points, max_devs):
        next_point = start_point
        data.append([object_id, 1, next_point])

        for i in range(1, num_members[counter] + 1):
            in_between = False
            while not in_between:
                summand = 0.06 * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    data.append([object_id + i, 1, next_point + summand])
                    in_between = True

        for time in range(2, num_timestamps + 1):
            in_between = False
            while not in_between:
                summand = max_dev * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    next_point += summand
                    data.append([object_id, time, next_point])
                    in_between = True

            for i in range(1, num_members[counter] + 1):
                in_between = False
                while not in_between:
                    summand = 0.06 * random.uniform(-1, 1)
                    if 0 <= next_point + summand <= 1:
                        data.append([object_id + i, time, next_point + summand])
                        in_between = True
        object_id += num_members[counter] + 1

    for i in range(num_outliers):
        x = random.uniform(0, 1)
        for time in range(1, num_timestamps + 1):
            in_between = False
            while not in_between:
                summand = 0.3 * random.uniform(-1, 1)
                if 0 <= x + summand <= 1:
                    x += summand
                    data.append([object_id + i, time, x])
                    in_between = True

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def generate_2d_data(num_timestamps=8, num_outliers=2, num_rand_trans=0, cluster_outliers=True, save_path=os.path.join('generated_data', 'generated_data_outlier.csv')):
    # random.seed(10) # generated_data.csv & generated_data2.csv & generated_data_pakdd2.csv
    # random.seed(9) # generated_data_outlier.csv & generated_data_pakdd.csv
    # random.seed(1) # generated_data_pakdd_trans.csv & generated_data_pakdd_trans2.csv
    random.seed(7) # generated_data_ipmu_trans.csv

    # centroids = [[0.3, 0.15], [0.7, 0.43], [0.45, 0.73]]
    # centroids = [[0.3, 0.15], [0.7, 0.43], [0.5, 0.7]] # generated_data_pakdd_trans.csv & generated_data_pakdd_trans2.csv
    centroids = [[0.5, 0.7], [0.7, 0.45], [0.3, 0.15]]  # generated_data_ipmu_trans.csv
    # num_cl_members = [15, 20, 10] # generated_data_pakdd_trans.csv & generated_data_pakdd_trans2.csv
    num_cl_members = [4, 3, 5]  # generated_data_ipmu_trans.csv
    # num_cl_members = [14, 18, 14]

    data = []

    # object_id = 4 # generated_data_ipmu_trans.csv
    object_id = 1
    for centroid, num_members in zip(centroids, num_cl_members):
        for time in range(1, num_timestamps + 1):
            centroid[0] += 0.05 * random.uniform(-1, 1)
            centroid[1] += 0.05 * random.uniform(-1, 1)
            for i in range(num_members):
                x = centroid[0] + 0.1 * random.uniform(-1, 1)
                y = centroid[1] + 0.1 * random.uniform(-1, 1)
                data.append([object_id + i, time, x, y])
            if cluster_outliers:
                x = centroid[0] + 0.25 * random.uniform(-1, 1)
                y = centroid[1] + 0.25 * random.uniform(-1, 1)
                data.append([object_id + num_members, time, x, y])
        object_id += num_members
        if cluster_outliers:
            object_id += 1

    for i in range(num_outliers):
        for time in range(1, num_timestamps + 1):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append([object_id + i, time, x, y])

    # object_id = 1 # generated_data_ipmu_trans.csv
    for i in range(num_rand_trans):
        for time in range(1, num_timestamps + 1):
            random_index = round(random.uniform(0, 1) * (len(centroids) - 2))  # generated_data_pakdd_trans2.csv
            # random_index = round(random.uniform(0, 1) * (len(centroids) - 2)) + 1 # generated_data_pakdd_trans.csv
            # random_index = round(random.uniform(0, 1) * (len(centroids) - 1))
            # x = centroids[random_index][0] + 0.15 * random.uniform(-1, 1)
            # y = centroids[random_index][1] + 0.15 * random.uniform(-1, 1)
            x = centroids[random_index][0] + 0.1 * random.uniform(-1, 1)
            y = centroids[random_index][1] + 0.1 * random.uniform(-1, 1)
            data.append([object_id + i, time, x, y])

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1', 'feature2'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def generate_long_1d_data(num_timestamps=40, num_outliers=5, save_path=None):
    # random.seed(8) # mldm_long
    # random.seed(8)  # mldm_long_40
    random.seed(23)  # mldm_long_40new

    start_points = [0.85, 0.65, 0.4, 0.15] # mldm_long
    max_devs = [0.1, 0.05, 0.05, 0.1]
    num_members = [5, 7, 8, 6]

    least_time = 5

    data = []
    centroids = []

    object_id = 1
    counter = 0
    for start_point, max_dev in zip(start_points, max_devs):
        next_point = start_point
        data.append([object_id, 1, next_point])
        cur_centroids = [next_point]

        for i in range(1, num_members[counter] + 1):
            in_between = False
            while not in_between:
                summand = 0.03 * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    data.append([object_id + i, 1, next_point + summand])
                    in_between = True

        for time in range(2, num_timestamps + 1):
            in_between = False
            while not in_between:
                summand = max_dev * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    next_point += summand
                    data.append([object_id, time, next_point])
                    in_between = True
                    cur_centroids.append(next_point)

            for i in range(1, num_members[counter] + 1):
                in_between = False
                while not in_between:
                    summand = 0.03 * random.uniform(-1, 1)
                    if 0 <= next_point + summand <= 1:
                        data.append([object_id + i, time, next_point + summand])
                        in_between = True
        centroids.append(cur_centroids)
        object_id += num_members[counter] + 1

    # one completely random outlier
    x = random.uniform(0, 1)
    for time in range(1, num_timestamps + 1):
        in_between = False
        while not in_between:
            summand = 0.1 * random.uniform(-1, 1)
            if 0 <= x + summand <= 1:
                x += summand
                data.append([object_id, time, x])
                in_between = True
    object_id += 1

    for i in range(num_outliers-1):
        cluster_time = 0
        next_cluster_id = round((len(start_points) - 1) * random.uniform(0, 1))
        last_cluster = next_cluster_id

        for time in range(1, num_timestamps + 1):
            if cluster_time >= least_time:
                next_cluster_id = round((len(start_points) - 1) * random.uniform(0, 1))
                while not x - 0.2 <= centroids[next_cluster_id][time - 1] <= x + 0.2:
                    next_cluster_id = round((len(start_points) - 1) * random.uniform(0, 1))

            # cluster_time = 0
            if next_cluster_id != last_cluster:
                last_cluster = next_cluster_id
                cluster_time = 0
            else:
                cluster_time += 1

            x = centroids[next_cluster_id][time - 1]

            in_between = False
            while not in_between:
                summand = 0.06 * random.uniform(-1, 1)
                if 0 <= x + summand <= 1:
                    x += summand
                    data.append([object_id + i, time, x])
                    in_between = True
                    cluster_time += 1

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data

def generate_1d_example_data(num_timestamps=8, num_outliers=1, save_path=None):
    random.seed(1)

    start_points = [0.8, 0.55, 0.25]
    max_devs = [0.03, 0.01, 0.05]
    num_members = [6, 6, 7]

    data = []
    centroids = []

    object_id = 1
    counter = 0
    for start_point, max_dev in zip(start_points, max_devs):
        cur_centroids = [start_point]
        next_point = start_point
        data.append([object_id, 1, next_point])

        for i in range(1, num_members[counter] + 1):
            in_between = False
            while not in_between:
                summand = 0.06 * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    data.append([object_id + i, 1, next_point + summand])
                    in_between = True

        for time in range(2, num_timestamps + 1):
            in_between = False
            while not in_between:
                summand = max_dev * random.uniform(-1, 1)
                if 0 <= next_point + summand <= 1:
                    next_point += summand
                    data.append([object_id, time, next_point])
                    in_between = True
                    cur_centroids.append(next_point)

            for i in range(1, num_members[counter] + 1):
                in_between = False
                while not in_between:
                    summand = 0.06 * random.uniform(-1, 1)
                    if 0 <= next_point + summand <= 1:
                        data.append([object_id + i, time, next_point + summand])
                        in_between = True
        object_id += num_members[counter] + 1
        centroids.append(cur_centroids)

    for i in range(num_outliers):
        for time in range(1, num_timestamps + 1):
            random_index = round(random.uniform(0, 1) * (len(centroids)-2))
            in_between = False
            while not in_between:
                if random_index == 0:
                    x = centroids[random_index][time-1] + 0.06 * random.uniform(-1, 0)
                elif random_index == 1:
                    x = centroids[random_index][time - 1] + 0.06 * random.uniform(0, 1)
                if 0 <= x <= 1:
                    data.append([object_id + i, time, x])
                    in_between = True

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def generate_five_data(num_timestamps=8, save_path=None):
    random.seed(1)
    num_members_corner = 10
    num_members_center = 20

    data = []
    for i in range(num_timestamps):
        object_id = 1
        for j in range(num_members_corner):
            # left lower corner
            x = 0.15 * random.uniform(0, 1)
            y = 0.15 * random.uniform(0, 1)
            data.append([object_id, i, x, y])
            object_id += 1

            # left upper corner
            x = 0.15 * random.uniform(0, 1)
            y = 1 - 0.15 * random.uniform(0, 1)
            data.append([object_id, i, x, y])
            object_id += 1

            # right lower corner
            x = 1 - 0.15 * random.uniform(0, 1)
            y = 0.15 * random.uniform(0, 1)
            data.append([object_id, i, x, y])
            object_id += 1

            # right upper corner
            x = 1 - 0.15 * random.uniform(0, 1)
            y = 1 - 0.15 * random.uniform(0, 1)
            data.append([object_id, i, x, y])
            object_id += 1
        for j in range(num_members_center):
            x = 0.5 + 0.1 * random.uniform(-1, 1)
            y = 0.5 + 0.1 * random.uniform(-1, 1)
            data.append([object_id, i, x, y])
            object_id += 1

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1', 'feature2'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def generate_random_five_data(num_timestamps=8, save_path=None):
    random.seed(1)
    num_members_corner = 10
    num_members_center = 20
    object_ids = np.arange(num_members_corner*4 + num_members_center)

    data = []
    for i in range(num_timestamps):
        random.shuffle(object_ids)
        oid = 0

        for j in range(num_members_corner):
            # left lower corner
            x = 0.15 * random.uniform(0, 1)
            y = 0.15 * random.uniform(0, 1)
            data.append([object_ids[oid], i, x, y])
            oid += 1

            # left upper corner
            x = 0.15 * random.uniform(0, 1)
            y = 1 - 0.15 * random.uniform(0, 1)
            data.append([object_ids[oid], i, x, y])
            oid += 1

            # right lower corner
            x = 1 - 0.15 * random.uniform(0, 1)
            y = 0.15 * random.uniform(0, 1)
            data.append([object_ids[oid], i, x, y])
            oid += 1

            # right upper corner
            x = 1 - 0.15 * random.uniform(0, 1)
            y = 1 - 0.15 * random.uniform(0, 1)
            data.append([object_ids[oid], i, x, y])
            oid += 1
        for j in range(num_members_center):
            x = 0.5 + 0.1 * random.uniform(-1, 1)
            y = 0.5 + 0.1 * random.uniform(-1, 1)
            data.append([object_ids[oid], i, x, y])
            oid += 1

    data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1', 'feature2'])
    if save_path:
        data.to_csv(save_path, index=False)
    return data


if __name__ == '__main__':
    # data = generate_1d_data(num_timestamps=8, num_outliers=5, save_path=os.path.join('generated_data', 'generated_data_mldm_long_40.csv'))
    # data = generate_2d_data(num_timestamps=8, num_outliers=0, num_rand_trans=3, cluster_outliers=False, save_path=os.path.join('generated_data', 'generated_data_pakdd_trans.csv'))
    # data = generate_long_1d_data(num_timestamps=40, num_outliers=4,
    #                         save_path=os.path.join('generated_data', 'generated_data_mldm_long_40new.csv'))
    # data = generate_2d_data(num_timestamps=4, num_outliers=0, num_rand_trans=3, cluster_outliers=False,
    #                         save_path=os.path.join('generated_data', 'generated_data_ipmu_trans.csv'))
    # data = generate_1d_example_data(num_timestamps=20, num_outliers=1, save_path=os.path.join('outlier_detection/generated_data', 'example_outlier.csv'))
    data = generate_random_five_data(num_timestamps=4, save_path='clustering/generated_data/generated_random_five_data.csv')
    # data = generate_long_1d_data(num_timestamps=40, num_outliers=0, save_path=os.path.join('clustering/generated_data', 'generated_data_long.csv'))
    print(data)