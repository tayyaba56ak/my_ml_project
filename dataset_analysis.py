import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Provide the correct path to the CSV files
df_normal = pd.read_csv('C:/Users/ASUS/Downloads/InSDN_DatasetCSV/Normal_data.csv')

# Display the first few rows of each dataset to confirm they are loaded
print("Normal Data:")
print(df_normal.head())

print("Shapes of dataset:", df_normal.shape)

print(df_normal.info())

print("Missing values per column:")
print(df_normal.isnull().sum())

print(df_normal.describe())

df_normal.fillna(df_normal.mean(), inplace=True) 

X= df_normal.iloc[:, :-1]
Y= df_normal.iloc[:, -1]

scaler = StandardScalar()
X_scale = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred)) 