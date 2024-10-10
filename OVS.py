
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


# load the OVS data 

df_ovs= pd.read_csv('C:/Users/ASUS/Downloads/InSDN_DatasetCSV/OVS.csv')

print(df_ovs.head()) # check first few rows of the data

print(df_ovs.shape) # how large the dataset is no.of rows/colums
print(df_ovs.columns) # cloumn headers

print(df_ovs.dtypes) # data type of each column
print(df_ovs.describe()) # get a summary of numerical columns

print(df_ovs.isnull().sum()) # check missing values in df

df_ovs_cleaned =df_ovs.dropna() # drop missing values

print(df_ovs.duplicated().sum()) 
df_ovs_cleaned =df_ovs_cleaned.drop_duplicates() # remove duplicates

scaler = StandardScaler() # prepare numerical data for ML
df_ovs_scaled= scaler.fit_transform(df_ovs_cleaned.select_dtypes(include=[float,int]))

X= df_ovs.iloc[:, :-1] # All rows, all columns except the last one
Y= df_ovs.iloc[:, -1]   # All rows, only the last column

print(X.dtypes)  # This will show the data types of each column
print(X.head())
non_numeric_columns = X.select_dtypes(include=['object']).columns

# Apply label encoding for categorical columns
le = LabelEncoder()
for col in non_numeric_columns:
    print(f"Encoding column: {col}")
    X[col] = le.fit_transform(X[col])

# Convert all columns to numeric (if needed)
X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
X.fillna(0, inplace=True)  # Handle NaN values by filling them with 0 or appropriate values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state=42) 

# choosing 10 model to train the data
#1 Logistic regression
model_lr =LogisticRegression()
model_lr.fit(X_train, Y_train)
y_pred_lr =model_lr.predict(X_test)

#model Evaluation
print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_pred_lr))
print(classification_report(Y_test, y_pred_lr)) 


#2 Decision Tree 
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,Y_train)
y_pred_dt = model_dt.predict(X_test)
print("Decision Tree Accuracy:" , accuracy_score(Y_test, y_pred_dt))
print(classification_report(Y_test, y_pred_dt))


#3 Random forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, Y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, y_pred_rf))
print(classification_report(Y_test, y_pred_rf)) 

#4 Support Vector Machine (SVM)
model_svm = SVC()
model_svm.fit(X_train, Y_train)
y_pred_svm =model_svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(Y_test, y_pred_svm))
print(classification_report(Y_test, y_pred_svm))

#5 k-Nearest Neighbors(KNN)
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, Y_train)
y_pred_knn = model_knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(Y_test, y_pred_knn))
print(classification_report(Y_test, y_pred_knn))

#6 Gradient Boosting:
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, Y_train)
y_pred_gb = model_gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(Y_test, y_pred_gb))
print(classification_report(Y_test, y_pred_gb)) 

#7 AdaBoosting
model_ab = AdaBoostClassifier()
model_ab.fit(X_train, Y_train)
y_pred_ab = model_ab.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(Y_test, y_pred_ab))
print(classification_report(Y_test, y_pred_ab))

#8 XGBoost 
model_xgb = XGBClassifier()
model_xgb.fit(X_train, Y_train)
y_pred_xgb = model_xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(Y_test, y_pred_xgb))
print(classification_report(Y_test, y_pred_xgb))

#9 Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train, Y_train)
y_pred_nb = model_nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(Y_test, y_pred_nb))
print(classification_report(Y_test, y_pred_nb)) 

#10 Extra trees Classifier
model_et = ExtraTreesClassifier()
model_et.fit(X_train, Y_train)
y_pred_et = model_et.predict(X_test)

print("Extra Trees Classifier Accuracy:", accuracy_score(Y_test, y_pred_et))
print(classification_report(Y_test, y_pred_et))




