# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the spam mail dataset (spam.csv).
3. Define hyperparameters and perform Grid Search for tuning.
4. Evaluate the model using accuracy, confusion matrix, and classification report.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: DEEKSHITHA S
Register Number: 212225100007


import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = "spam.csv"   # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

data = pd.read_csv(file_path, encoding=result['encoding'])

print(data.head())
print(data.info())
print(data.isnull().sum())

plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

x = data['v2'].values     # messages
y = data['v1'].values     # labels

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

y_pred = svc.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
```

## Output:
<img width="472" height="391" alt="image" src="https://github.com/user-attachments/assets/a4226c85-deab-4033-a803-86f21eeee9c8" />

<img width="1003" height="700" alt="image" src="https://github.com/user-attachments/assets/b519d1c7-914d-48f4-bd76-7220c9acb5af" />

<img width="549" height="391" alt="image" src="https://github.com/user-attachments/assets/6257c55b-a96b-4014-929b-dd76cf53e434" />

<img width="847" height="811" alt="image" src="https://github.com/user-attachments/assets/c0086d6c-8285-49c8-b756-23c390a3470a" />






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
