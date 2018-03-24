import pandas as pd
from sklearn.datasets import load_breast_cancer
import xgboost as xgb

from prediction_path import prediction_path

# load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
dtrain = xgb.DMatrix(df, label=y)

# train
params = {"max_depth": 10, "eta": 1}
bst = xgb.train(params, dtrain, 1)

# get prediction paths
x1 = df.iloc[67]
print(prediction_path(bst, x1, 0))
