import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class HyperParams:
  
  def __init__(self):    
    self.hp_lightGBM = {
      "objective" : "regression",
      "metric" : "rmse",
      "n_estimators": 1000,
      "num_leaves" : 30,
      "min_child_samples" : 30,
      "learning_rate" : 0.005,
      "verbosity" : -1
    }
    self.hp_CVlightGBM = {
      "objective" : "regression",
      "metric" : "rmse",
      "n_estimators": 1000,
      "num_leaves" : 30,
      "min_child_samples" : 30,
      "learning_rate" : 0.005,
      "verbosity" : -1
    }
    self.hp_randomForestRegressor = {
      "criterion" : "mse", #mae original 
      "max_depth" : 8,
      "max_leaf_nodes" : None,
      "min_impurity_split" : None,
      "min_samples_leaf" : 1,
      "min_samples_split" : 2,
      "min_weight_fraction_leaf" : 0.0,
      "n_estimators" : 500,
      "n_jobs" : -1,
      "random_state" : 0,
      "verbose" : 0,
      "warm_start" : False
    }
    self.hp_CVrandomForestRegressor = {
      "criterion" : ['mse'], #mae original 
      "max_depth" : 8,
      # "max_leaf_nodes" : None,
      # "min_impurity_split" : None,
      "min_samples_leaf" : 1,
      "min_samples_split" : 2,
      "min_weight_fraction_leaf" : [0.0],
      "n_estimators" : 500,
      "n_jobs" : [-1],
      # "random_state" : 0,
      "verbose" : [1],
      "warm_start" : [False]
    }

class AlgoML(HyperParams):
  
  def __init__(self):
    super().__init__()
    pass

  def lightGBM(self, obj_data):
    
    params = {
      "objective" : "regression",
      "metric" : "rmse",
      "n_estimators": 1000,
      "num_leaves" : 30,
      "min_child_samples" : 30,
      "learning_rate" : 0.005,
      "verbosity" : -1
    }

    #creating dataset for lgb
    lgb_dataFrameTrain = lgb.Dataset(obj_data["x_train"], label=obj_data["y_train"])
    lgb_dataFrameValidation = lgb.Dataset(obj_data["x_test"], label=obj_data["y_test"])

    ml = lgb.train(params, lgb_dataFrameTrain, 8000, valid_sets=[lgb_dataFrameValidation], early_stopping_rounds=100, verbose_eval=100)

    return ml

  def randomForestRegressor(self, obj_data = None, p_xtrain = None, p_ytrain = None):
    regressor = RandomForestRegressor(**self.hp_randomForestRegressor)

    if(obj_data == None):
      return regressor.fit(p_xtrain, p_ytrain)
    else:
      return regressor.fit(obj_data["x_train"], obj_data["y_train"])

  def gridSearchCV(self, model, metrics=None, p_cv=2):
    # first set self.hp_CV<the model> as desire
    # second insert into param @model the model that desire to do the gridSearchCV 

    model_options = {
      "lightGBM": [lgb.LGBMRegressor(), self.hp_CVlightGBM],
      "randomForestRegressor": [RandomForestRegressor(), self.hp_CVrandomForestRegressor]
    }

    if(metrics == None):
      metrics = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']

    grid = GridSearchCV(model_options[model][0], param_grid=model_options[model][1], scoring=metrics, verbose=100, refit='neg_mean_squared_error', return_train_score=False, cv=p_cv)
    # grid = GridSearchCV(model_options[model][0], param_grid=model_options[model][1], verbose=100, return_train_score=False)
    return grid