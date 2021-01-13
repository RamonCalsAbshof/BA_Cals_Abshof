import numpy as np
from sklearn.linear_model import LinearRegression

class linear_regression(object):




    def __init__(self, settings, prediction_result, data):
        self._settings = settings
        self._prediction_result = prediction_result
        self._data = data

    def create_multidim_model(self, object, time, prediction_data):
        result_dict = dict()
        result_dict['ObjectID'] = object
        prediction_data = prediction_data[prediction_data['ObjectID']==object]

        object = object
        y = prediction_data[self._settings['feature_renames']].to_numpy()
        x = prediction_data['Time'].to_numpy().reshape(-1, 1)
        model = LinearRegression().fit(x,y)
       # print('coefficient of determination:', model.score(self._x, self._y))
       # print('intercept:', model.intercept_)
       # print('slope:', model.coef_)

        # predict

        for feature in self._settings['feature_renames']:
            result_dict[feature] = \
            self._data[(self._data['ObjectID'] == object) & (self._data['Time'] == time)][feature].values[0]

        for i in range(0, len(self._settings['feature_renames'])):
            result_dict['p_' + self._settings['feature_renames'][i]] = (model.intercept_ + np.sum(model.coef_ * time, axis=1))[i]

        #print(model.predict(x))

        self._prediction_result = self._prediction_result.append(result_dict, ignore_index=True)

    def create_model(self, object, time, prediction_data):
        result_dict = dict()
        result_dict['ObjectID'] = object
        prediction_data = prediction_data[prediction_data['ObjectID']==object]
        for feature in self._settings['feature_renames']:
            object = object
            y = prediction_data[feature].to_numpy()
            x = prediction_data['Time'].to_numpy().reshape(-1, 1)
            model = LinearRegression().fit(x,y)
           # print('coefficient of determination:', model.score(self._x, self._y))
           # print('intercept:', model.intercept_)
           # print('slope:', model.coef_)

            # predict


            result_dict[feature] = \
            self._data[(self._data['ObjectID'] == object) & (self._data['Time'] == time)][feature].values[0]
            result_dict['p_' + feature] = model.intercept_ + model.coef_[0] * time
            print(model.predict(x))

        self._prediction_result = self._prediction_result.append(result_dict, ignore_index=True)


    def get_prediction_frame(self):
        return self._prediction_result