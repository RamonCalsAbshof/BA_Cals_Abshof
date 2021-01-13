import nibabel
import numpy as np
import pandas as pd

orig_path = '/home/data/Rockland_Data/'
csv_name = 'participants_std.csv'
norm_path = 'Data_Cropped/'
gray_path = 'Data_Cropped_Gray/'
white_path = 'Data_Cropped_White/'
paths = [norm_path, gray_path, white_path]


def get_filenames(label_file, deli=','):
    filename_list, _ = zip(*np.genfromtxt(label_file, dtype=None, delimiter=deli, skip_header=1, usecols=(0, 1),
                                          unpack=True, encoding='utf8'))
    return filename_list


def main():
    filenames = get_filenames(orig_path + csv_name)
    data = []
    for filename in filenames:
        features = []
        for path in paths:
            img = nibabel.load(orig_path + path + filename).get_data()
            img += 1
            img = np.moveaxis(img, -1, 0)
            tmp_features = []
            for slice in img:
                tmp_features.append(np.count_nonzero(slice))
            features.append(tmp_features)
        features = np.array(features)
        features = np.transpose(features)
        time_counter = 1
        for item in features:
            data.append(list(np.concatenate((np.array([filename, time_counter]), list(item)))))
            time_counter += 1
    feature_frame = pd.DataFrame(data, columns=['Filename', 'Time', 'Total', 'Gray', 'White'])
    feature_frame.to_csv(orig_path + 'mri_features.csv')


if __name__ == '__main__':
    main()
