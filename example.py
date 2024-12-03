import numpy as np  
import pandas as pd 
import pickle 
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier 


data = pd.read_csv('diabetes.csv')  

X = data.drop('Outcome', axis=1)  
y = data['Outcome'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

model = KNeighborsClassifier()  
model.fit(X_train, y_train) 


score = model.score(X_test, y_test)  
print(f'Model score: {score}') 

with open('model.pkl', 'wb') as file:  
    pickle.dump(model, file)  


X_test.to_csv('X_test.csv', index=False) 
y_test.to_csv('y_test.csv', index=False) 
