import pandas as pd
import numpy as np
from sklearn import tree

data0 = pd.read_csv("train.csv")
data = data0.dropna()
data["sex_trans"] = [1 if value == "male" else 0 for value in data["Sex"]]
embarkment_map = {
    "C": 0,
    "Q": 1,
    "S": 2,
}
data["embarked_trans"] = [embarkment_map[value] for value in data["Embarked"]]
x = data[["sex_trans", "Pclass", "Age", "SibSp", "Parch", "embarked_trans"]].values
y = data["Survived"].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

pred_x = clf.predict(x)
print(f"The accuracy is {np.round(np.mean(pred_x==y), 2)}")