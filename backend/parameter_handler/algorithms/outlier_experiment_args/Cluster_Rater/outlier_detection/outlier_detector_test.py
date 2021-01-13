import pandas as pd
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector


def test():
    test_data = pd.DataFrame(data={'ObjectID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                   'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                                   'Cluster': [1, 3, 5, 1, 3, 5, 1, 4, 5, 2, 4, 6, 2, 4, 6, 2, 3, -1]})
    outlier_detector = OutlierDetector(test_data)
    rating = outlier_detector.calc_outlier_rating()
    print(rating)


test()