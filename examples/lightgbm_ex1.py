import lightgbm as lgb
import pandas as pd
from sklearn.datasets import load_breast_cancer

from prediction_path import prediction_path

# load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# train
clf = lgb.LGBMClassifier(num_leaves=100, n_estimators=1)
clf.fit(df, y)

# get prediction path and validate
idx = 24
x = df.iloc[idx]
print(prediction_path(clf, x, 0))
print(clf.apply(df[idx: idx + 1]))
