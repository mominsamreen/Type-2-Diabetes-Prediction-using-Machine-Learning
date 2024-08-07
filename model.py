import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data=pd.read_csv('./diabetes.csv')
data['Pregnancies'].fillna(data['Pregnancies'].mean(),inplace=True)
data['Glucose'].fillna(data['Glucose'].mean(),inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(),inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].mean(),inplace=True)
data['Insulin'].fillna(data['Insulin'].mean(),inplace=True)
data['BMI'].fillna(data['BMI'].mean(),inplace=True)
data['DiabetesPedigreeFunction'].fillna(data['DiabetesPedigreeFunction'].mean(),inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
data = np.array(data)
X=data[1:,0:-1]
y=data[1:,-1]
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
diabetes=RandomForestClassifier(random_state=10)
print(len(X_train[0]))
print((y_train))
diabetes.fit(X_train, y_train)
pickle.dump(diabetes,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))