import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class Handler:

  def __init__(self):
    pass

  def get_sample(self, df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]

  def create_test_from_train(self, dfTrain, target, split_size = 0.30, p_random = 0):
    """
    @param dftrain -- Pandas.DataFrame -- set used for train 
    @param target -- Pandas.Series -- dependent variable we want to predict
    @param split_size -- float -- percentage will be splited our set i.e 30% == 30% for test and 70% for train
    @param random -- int -- percentage to random
    @return obj_return -- dictionary -- dictionary storing our set of train and test 
    """

    X_train, X_test, Y_train, Y_test = train_test_split(dfTrain, target, test_size=split_size, random_state = p_random)

    obj_return = {
      "x_train": X_train,
      "y_train": Y_train,
      "x_test": X_test,
      "y_test": Y_test
    }

    return obj_return

  def take_dataframe_with_features(self, df, features):
    """
    @param df -- Pandas.DataFrame 
    @param features -- List -- List of strings containing the features to select
    @return Pandas.DataFrame -- the new dataframe only containing the selected features 
    """

    return df[features]

  def RFE(self, df, target = None, qnt_features=5):
    """
    @param df -- Pandas.DataFrame -- dataframe with target into or apart
    @param target -- String | pd.Series -- the dependent variable, if was passed a string it will take of df if 
                                          not it will assume that already passed our target with their values as pd.Series
    @qnt_features -- int -- total of features to take
    @return list< list result features, rfe object > -- the features selected by RFE and the RFE object 
    """

    x = None
    y = None

    if(type(target) == type(str())):
      x = df.drop(target, axis=1)
      y = df[target]
    else:
      x = df.drop(target.name, axis=1)
      y = target
    
    _estimator = LinearRegression()
    
    _rfe = RFE(estimator=_estimator, n_features_to_select=qnt_features)
    _rfe.fit(x, y)
    
    return [ list(x.loc[:, _rfe.support_].columns), _rfe]

  def get_onlyNumeric(self, df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)

    return newdf

  def PCA(self, df, number_components = 0.95):
    _pca = PCA(n_components=number_components)
    x_transformed = _pca.fit_transform(df)
    
    return x_transformed.shape[1]