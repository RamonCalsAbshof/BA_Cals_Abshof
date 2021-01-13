import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

step_size = 9
compress_timestamps = False
normalize = False
plot = True

features_csv = 'mri_features_steps9_norm.csv'
feature_names = ['Total', 'Gray', 'White']


def plot_data(data, feature='Gray', num_mri=30):
    filenames = data['Filename'].unique()
    ax = plt.gca()
    counter = 1
    for filename in filenames:
        data[data['Filename'] == filename].plot(kind='line', x='Time', y=feature, ax=ax)
        counter += 1
        if counter > num_mri:
            break
    plt.show()


def normalize_features(data):
    for feature in feature_names:
        f1 = data[[feature]].values.astype(float)

        min_max_scaler_dep = preprocessing.MinMaxScaler()
        f1_scaled = min_max_scaler_dep.fit_transform(f1)

        data[feature] = f1_scaled
    return data


def partition_time(data):
    filenames = data['Filename'].unique()
    new_data = pd.DataFrame(columns=['Filename', 'Time', 'Total', 'Gray', 'White'])
    for filename in filenames:
        f_data = data[data['Filename'] == filename]
        counter = 0
        while counter < round(float(179) / float(step_size)):
            step = f_data[(f_data['Time'] > counter * step_size) & (f_data['Time'] <= (counter + 1) * step_size)].sum(
                axis=0)
            new_data = new_data.append(pd.DataFrame({'Filename': [filename],
                                                     'Time': [counter + 1],
                                                     'Total': [step['Total']],
                                                     'Gray': [step['Gray']],
                                                     'White': [step['White']]}))
            counter += 1
    return new_data


def main():
    data = pd.read_csv(features_csv)
    if compress_timestamps:
        data = partition_time(data)
        data.to_csv('mri_features_steps9.csv', index=False)
    if normalize:
        data = normalize_features(data)
        data.to_csv('mri_features_steps9_norm.csv', index=False)
    if plot:
        plot_data(data)


if __name__ == '__main__':
    main()