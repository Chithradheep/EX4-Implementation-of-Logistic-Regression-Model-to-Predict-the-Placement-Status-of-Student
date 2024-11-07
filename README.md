# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess Data: Drop unnecessary columns and encode categorical features into numerical
form.
2. Split Data: Divide the dataset into training and testing sets using train_test_split().
3. Train Model: Use LogisticRegression to train the model on the training data.
4. Evaluate Model: Predict placement status and calculate accuracy, confusion matrix, and
classification report.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Chithradheep R
RegisterNumber:  2305002003
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
x=data1.iloc[:,:-1]
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
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True]
cm_display.plot()
```

## Output:
![Screenshot 2024-11-07 094454](https://github.com/user-attachments/assets/4301055b-a590-494f-8eec-51dc164f0b77)
![Screenshot 2024-11-07 094522](https://github.com/user-attachments/assets/541d3275-9efd-4183-8410-9dbe0af050b4)
![Screenshot 2024-11-07 094551](https://github.com/user-attachments/assets/8ade8c7b-35d2-433f-9406-b7784ade9e70)
![Screenshot 2024-11-07 094620](https://github.com/user-attachments/assets/72e2737b-ced8-4b01-a81c-5e427921bbfd)
![Screenshot 2024-11-07 094640](https://github.com/user-attachments/assets/020376a7-458c-4f29-88b4-6b5211e9b318)
![Screenshot 2024-11-07 094658](https://github.com/user-attachments/assets/283000ba-3451-46be-9aef-d040477c3840)
![Screenshot 2024-11-07 094725](https://github.com/user-attachments/assets/11bce8a8-a5b0-4b6c-b36d-a83be239e6bd)
![Screenshot 2024-11-07 094752](https://github.com/user-attachments/assets/75bd45ea-1e8b-40e6-af27-50164d45164d)
![Screenshot 2024-11-07 094817](https://github.com/user-attachments/assets/ac5d82de-5cb8-442a-bb84-1a0fe8bd71a9)
![Screenshot 2024-11-07 094839](https://github.com/user-attachments/assets/7120989c-426c-4bf8-9a86-2e1d3731e015)
![image](https://github.com/user-attachments/assets/31b9ad79-b11e-4690-923b-f202ce839b4d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
