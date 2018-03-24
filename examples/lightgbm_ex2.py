import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from prediction_path import prediction_path

# load data
data = load_breast_cancer()

# use only the first 3 columns to demonstrate different properties
df = pd.DataFrame(data.data[:, 0:3], columns=data.feature_names[0:3])
y = data.target

# insert missing values
idx = df["mean radius"] < 12
df.loc[idx, "mean radius"] = np.nan

# convert into a categorical feature
idx2 = df["mean texture"] < 18
df.loc[idx2, "mean texture"] = 0
df.loc[~idx2, "mean texture"] = 1

# training
train_set = lgb.Dataset(df, label=y)
params = {"boosting_type": "gbdt", "objective": "binary", "num_leaves": 50}
gbdt = lgb.train(params, train_set, num_boost_round=1, categorical_feature=[1])

# get prediction path
i = 3
x = df.iloc[i]
print(prediction_path(gbdt, x, 0))
