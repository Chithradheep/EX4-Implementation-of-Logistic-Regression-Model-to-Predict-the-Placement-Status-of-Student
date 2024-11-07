# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, handle missing values, and encode categorical variables.
2. Select relevant features that impact placement status.
3. Split the data into training and testing sets.
4. Use a Logistic Regression model to train on the training data.
5. Predict placement status on the test set.
6. Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Chithradheep R
RegisterNumber: 2305002003

import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,: -1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification rate:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
*/
```

## Output:
![Screenshot 2024-11-07 094454](https://github.com/user-attachments/assets/7ed931f8-ddab-47e9-b765-1aaf57605cda)
![Screenshot 2024-11-07 094522](https://github.com/user-attachments/assets/6bf5d34f-3646-4564-abcc-dff062585328)
![Screenshot 2024-11-07 094551](https://github.com/user-attachments/assets/4e0d7eb4-cf63-4c19-a4b4-03cab7b0bcd2)
![Screenshot 2024-11-07 094620](https://github.com/user-attachments/assets/4aa0036d-c02d-4730-b7a5-8bbb36f8e707)
![Screenshot 2024-11-07 094640](https://github.com/user-attachments/assets/8fe53671-915c-4a11-a495-27050d296561)
![Screenshot 2024-11-07 094658](https://github.com/user-attachments/assets/8259fa3f-813c-4f3b-a515-debc9e8b64eb)
![Screenshot 2024-11-07 094725](https://github.com/user-attachments/assets/1f66d6ba-c78f-4fac-b2d3-397b127ca37a)
![Screenshot 2024-11-07 094752](https://github.com/user-attachments/assets/82762290-311a-4bdf-97b4-b4329e18b2fc)
![Screenshot 2024-11-07 094817](https://github.com/user-attachments/assets/25adfe66-8479-48e6-a494-6120aaccc0f1)
![Screenshot 2024-11-07 094839](https://github.com/user-attachments/assets/9694ba83-a97e-4410-9512-6c7665229a45)
![Screenshot 2024-11-07 094903](https://github.com/user-attachments/assets/8845ba14-5c3d-4537-aba3-91ec17f06a74)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
