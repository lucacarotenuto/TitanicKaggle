import pandas as pd
import numpy as np
import seaborn as sns
import string
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
all = pd.concat([train, test]).reset_index(drop=True)

# fix missing values
all['Age'] = all.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
all['Fare'] = all['Fare'].fillna(all['Fare'].mean())
all['Embarked'] = all['Embarked'].fillna('S')
#all['Cabin'] = all['Cabin'].fillna('X')

# is alone feature
all['isalone'] = all['SibSp'] + all['Parch']
all.loc[all['isalone'] > 0, 'isalone'] = 'No'
all.loc[all['isalone'] == 0, 'isalone'] = 'Yes'

# title feature, name the titles with less than 10 appearances to title "Other"
all["Title"] = all["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (all["Title"].value_counts() < 10)
all["Title"] = all["Title"].apply(lambda x: "Other" if title_names.loc[x] == True else x)

# other features
all["Deck"] = all["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "Missed")
all["Fsize"] = all["SibSp"] + all["Parch"]
all['FarePerPerson'] = all['Fare'] / (all['Fsize']+1)
all['Ticket_Frequency'] = all.groupby('Ticket')['Ticket'].transform('count')

# transform to numerical
enc = LabelEncoder()
for col in ['Sex', 'Pclass', 'isalone', 'Embarked', 'Title', 'Deck', 'Ticket']:
    print(col)
    all[col] = enc.fit_transform(all[col])
    print(enc.classes_)

# non-ordinal one hot encoding
cat_features = ['Sex', 'isalone', 'Embarked', 'Title', 'Deck']
encoded_features = []
for feature in cat_features:
    encoded_feat = OneHotEncoder().fit_transform(all[feature].values.reshape(-1, 1)).toarray()
    n = all[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = all.index
    encoded_features.append(encoded_df)

all = pd.concat([all, *encoded_features[:6]], axis=1)
all = all.drop(["Name", "Cabin","Ticket","PassengerId"], axis=1)
all = all.drop(cat_features, axis=1)

# reconstruct train and test
train = all.iloc[:891]
test = all.iloc[891:]
X_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']

X_train = pd.DataFrame(data=train)
X_train = X_train.drop('Survived', axis=1)
y_train = pd.DataFrame(columns=['Survived'], data=train)
X_test = pd.DataFrame(data=test)
X_test = X_test.drop('Survived', axis=1)

# Grid Search with K-fold CV Random Forest
rf = RandomForestClassifier(oob_score=False, random_state=1, n_jobs=-1)
param_grid = {"max_depth":[9,11,13],
             "min_samples_split": [10,12,14], "n_estimators": [450,550,650]}
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy',
                  cv=5, n_jobs=-1, verbose=2)

gs = gs.fit(X_train, y_train.values.ravel())
print(gs.best_score_)
print(gs.best_params_)

# track memory usage
joblib.dump(gs.best_estimator_.estimators_[0], "first_tree_from_RF.joblib")
print(f"Single tree size: {np.round(os.path.getsize('first_tree_from_RF.joblib') / 1024 / 1024, 2) } MB")
joblib.dump(gs.best_estimator_, "RandomForest_100_trees.joblib")
print(f"Random Forest size: {np.round(os.path.getsize('RandomForest_100_trees.joblib') / 1024 / 1024, 2) } MB")
print(X_train.info())

# feature importance
importances = gs.best_estimator_.feature_importances_
plt.figure(figsize=(15, 20))
sns.barplot(y=gs.best_estimator_.feature_names_in_, x=importances)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title('Random Forest Classifier Feature Importance', size=20)
plt.savefig('myfile.png', bbox_inches = "tight")
plt.show()

# train again on validation set for confusion matrix
X_train_conf, X_valid_conf = train_test_split(X_train, test_size=0.2, random_state=42)
y_train_conf, y_valid_conf = train_test_split(y_train, test_size=0.2, random_state=42)
estimator_conf = gs.best_estimator_.fit(X_train_conf, y_train_conf)
y_pred_conf = estimator_conf.predict(X_valid_conf)
conf = confusion_matrix(y_valid_conf, y_pred_conf)
tn, fp, fn, tp = conf.ravel()
print('Accuracy ' + str((tp+tn) / (tp+tn+fp+fn)))
print('Precision' + str(tp/(tp+fp)))
print('Recall' + str(tp/(tp+fn)))
ConfusionMatrixDisplay.from_predictions(y_valid_conf, y_pred_conf, colorbar=False, cmap='copper')
plt.show()

# predict test set for submission
y_test = gs.predict(X_test)
res = pd.DataFrame(data={'PassengerId': [i for i in range(892,1310)], 'Survived' : y_test}).astype(int)
print(res)
res.to_csv('result.csv', index=False)