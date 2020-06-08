import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Refine:

    def __init__(self):
      pass

    def clear_null(self, df):
      """
      @param df -- Pandas.DataFrame
      @return -- Pandas.DataFrame
      """
      return df.dropna(inplace=False)

    def normalize(self, df):
      """
      @param df -- Pandas.DataFrame
      @return -- MinMaxScaler object
      """
      scaler = MinMaxScaler(copy=True)
      return scaler.fit(df)
    
    def standardize(self, df):
      """
      @param df -- Pandas.DataFrame
      @return -- StandardScaler object
      """
      scaler = StandardScaler(copy=True)
      return scaler.fit(df)

    def imputer_na(self, df, p_strategy, p_missing_value = np.nan):
      imputer = SimpleImputer(missing_values=np.nan, strategy=p_strategy, copy=True)
      return imputer.fit_transform(df)

    def filter_columns_percent_null(self, df, percent = 0.5):
      columns_null = [i for i in df.columns if df[i].isnull().any()]
      values_null = df[columns_null].isna().sum()

      filter_columns = [x for x in columns_null if (values_null[x]/df.shape[0]) < percent]
      return filter_columns