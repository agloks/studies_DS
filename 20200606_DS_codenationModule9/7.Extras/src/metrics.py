import pandas as pd
from sklearn.metrics import mean_squared_error, \
                            mean_absolute_error, \
                            confusion_matrix, \
                            classification_report, \
                            roc_auc_score, \
                            average_precision_score
import numpy as np

class Metrics:
    def __init__(self):
        pass

    def calculate_regression(self, y_true, y_pred):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''
        mean_abs_err = mean_absolute_error(y_true, y_pred)
        mean_sqr_err = mean_squared_error(y_true, y_pred)
        return {'mean_abs_err' : mean_abs_err, 
                'mean_sqr_err' : mean_sqr_err,
                'r_mean_sqr_err': np.sqrt(mean_sqr_err)}

    def calculate_classification(self, y_true, y_pred):
        '''
        Calculate the metrics from a classification problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''
        rocauc_score = roc_auc_score(y_true, y_pred)
        confu_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        avg_precision_score = average_precision_score(y_true, y_pred)

        print(class_report)

        print(f"""
        ----------------------------------------------------------------------
          True Negatives = {confu_matrix[0][0]}             |    False Positives = {confu_matrix[0][1]}
          False Negatives = {confu_matrix[1][0]}            |    True Positives = {confu_matrix[1][1]}
        ----------------------------------------------------------------------
            """)

        print(f"roc_auc_score = {rocauc_score}")
        print(f"avg_precision_score = {avg_precision_score}")