# -*- coding: utf-8 -*-
"""
Department of Computer Engineering, LGNSCOE NASHIK

 Implement Naive Bayes Algorithm for classification of Pima Indians Diabetes dataset.
      1)	Load the data from CSV file and split it into training and test datasets.
      2)	Summarize the properties in the training dataset so that we can calculate Probabilities and make predictions.
      3)	Classify samples from a test dataset and a summarized training dataset.
"""

#importing libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset = pd.read_csv('diabetes.csv')

#split dataset
print(dataset.describe())
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

#view coorelation
sns.heatmap(X.corr(), annot = True)

#replace zero values with median
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    X[column] = X[column].replace(0, np.NaN)
    mean = int(X[column].mean(skipna=True))
    X[column] = X[column].replace(np.NaN, mean)
    
    
#feature extraction
## Var[X] = p(1-p)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_filtered = sel.fit_transform(X)

print(X.head(1))
print(X_filtered[0])
#DiabetesPedigreeFunction was dropped
X = X.drop('DiabetesPedigreeFunction', axis=1)

top_4_features = SelectKBest(score_func=chi2, k=4)
X_top_4_features = top_4_features.fit_transform(X, y)
print(X.head())
print(X_top_4_features)
X = X.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1)


#split dataset into 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#implement classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predict test result
y_pred = classifier.predict(X_test)

#evaluate model
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



''' out put 
satish@satish-PC:~/Desktop/Asg2/diabetes$ python diab.py 
       Pregnancies     Glucose     ...             Age     Outcome
count   768.000000  768.000000     ...      768.000000  768.000000
mean      3.845052  120.894531     ...       33.240885    0.348958
std       3.369578   31.972618     ...       11.760232    0.476951
min       0.000000    0.000000     ...       21.000000    0.000000
25%       1.000000   99.000000     ...       24.000000    0.000000
50%       3.000000  117.000000     ...       29.000000    0.000000
75%       6.000000  140.250000     ...       41.000000    1.000000
max      17.000000  199.000000     ...       81.000000    1.000000

[8 rows x 9 columns]
   Pregnancies  Glucose  BloodPressure ...    BMI  DiabetesPedigreeFunction  Age
0            6    148.0           72.0 ...   33.6                     0.627   50

[1 rows x 8 columns]
[  6.  148.   72.   35.  155.   33.6  50. ]
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  Age
0            6    148.0           72.0           35.0    155.0  33.6   50
1            1     85.0           66.0           29.0    155.0  26.6   31
2            8    183.0           64.0           29.0    155.0  23.3   32
3            1     89.0           66.0           23.0     94.0  28.1   21
4            0    137.0           40.0           35.0    168.0  43.1   33
[[  6. 148. 155.  50.]
 [  1.  85. 155.  31.]
 [  8. 183. 155.  32.]
 ...
 [  5. 121. 112.  30.]
 [  1. 126. 155.  47.]
 [  1.  93. 155.  23.]]
[[116  14]
 [ 30  32]]
0.5925925925925926
0.7708333333333334
satish@satish-PC:~/Desktop/Asg2/diabetes$ 
'''
