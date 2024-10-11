import pandas as pd  
import unittest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

df =pd.read_csv('C:/Users/ASUS/Downloads/InSDN_DatasetCSV/metasploitable-2.csv')

# check data
print(df.head()) # Shows the first 5 rows of the dataset 

#clean data 
df= df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'])  # Remove unnecessary columns 

# Remove duplicate rows
df = df.drop_duplicates()

#check if there are any missing values in the data
print(df.isnull().sum()) # Shows how many missing values each column has 
df = df.dropna()  # This will remove any rows that have missing values


label_encoder =LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])  # Encode the labels as numbers

X = df.drop(columns=['Label']) # this is our input data
y = df['Label'] # this is what we want to predict 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Applying 10 models
#1 LogisticRegression 

model = LogisticRegression()
model.fit(X_train,y_train) # Train it with the training model
y_pred = model.predict(X_test) # Predict on the testing data
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')

#2 Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Decision Tree Accuracy: {accuracy}') 

#3 Random Forest 
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Random Forest Accuracy: {accuracy}') 

#4 Supoort Vector Machine 
model = SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'SVM accuracy: {accuracy}')

# 5 Naive Bayes 
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Naive Bayes Accuracy: {accuracy}')

#6 Gradient Boosting classifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Gradient Boosting Accuracy: {accuracy}')

#7 AdaBoost 
model = AdaBoostClassifier(algorithm="SAMME")
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Ada Boost Accuracy: {accuracy}') 

#8 K-nearest Neighbor (KNN) 
model = KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'KNN Accuracy: {accuracy}')

#9 Bagging 
model = BaggingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Bagging Accuracy: {accuracy}') 

#10 Neural Network (MLP Classifier)
model = MLPClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Neural Network Accuracy: {accuracy}')

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Load the CSV file and drop unnecessary columns
        self.df = pd.read_csv('C:/Users/ASUS/Downloads/InSDN_DatasetCSV/metasploitable-2.csv')
        self.df_cleaned = self.df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'])

    def test_drop_columns(self):
        # Check if 'Flow ID' column was successfully dropped
        self.assertNotIn('Flow ID', self.df_cleaned.columns)

    def test_label_encoding(self):
        # Apply label encoding to the 'Label' column and check the number of unique labels
        le = LabelEncoder()
        self.df_cleaned['Label'] = le.fit_transform(self.df_cleaned['Label'])
        self.assertEqual(self.df_cleaned['Label'].nunique(), len(le.classes_))

if __name__ == '__main__':
    unittest.main()




