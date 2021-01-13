import pandas as pd
from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.gps_dataset_provider import GPSDataset
from Cluster_Rater.economy_dataset_provider import EconomyDataset
from sklearn import preprocessing


def normalize(data, feature):
    f1 = data[[feature]].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    f1_scaled = min_max_scaler_dep.fit_transform(f1)

    data[feature] = f1_scaled
    return data


def get_dev_data(data, DATASET):
    dev_data = pd.DataFrame(columns=list(data.columns))
    rics = data['ObjectID'].unique().tolist()
    for ric in rics:
        for time in range(DATASET['start_year'], DATASET['end_year']):
            cur = data[(data['ObjectID'] == ric) & (data['Time'] == time)]
            next = data[(data['ObjectID'] == ric) & (data['Time'] == time + 1)]
            if len(cur) > 0 and len(next) > 0:
                dev_dict = {'ObjectID': ric, 'Time': time + 1}
                for feature in DATASET['features']:
                    dev_dict[feature] = next[feature].item() - cur[feature].item()
                dev_data = dev_data.append(dev_dict, ignore_index=True)
    return dev_data


def load_data(DATASET):
    data = None
    if DATASET['Name'] == 'flights':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()
        time_points = data['Time'].unique()
        time_points.sort()
        for i in range(0, len(time_points)):
            data.loc[data['Time'] == time_points[i], 'Time'] = i + 1

    elif DATASET['Name'] == 'financial':
        db = MongoDB(DATASET)
        data = db.get_financial_data(True)
        # cast strings into numerics
        for feature in DATASET['feature_renames']:
            data[feature] = pd.to_numeric(data[feature])
        data['Time'] = pd.to_numeric(data['Time'])

    elif DATASET['Name'] == 'generated' or DATASET['Name'] == 'diabetes':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()

    elif DATASET['Name'] == 'gps_data':
        gps_data = GPSDataset(DATASET)
        data = gps_data.get_data()

    elif DATASET['Name'] == 'economy':
        economy_data = EconomyDataset(DATASET)
        data = economy_data.get_data()
    return data