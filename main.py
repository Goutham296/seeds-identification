import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler# Sample seed data with some duplicates
data = {
    'area': [15.2, 14.5, 15.2, 13.8, 16.5, 15.2],  # duplicate area
    'perimeter': [14.1, 13.2, 14.1, 12.8, 15.0, 14.1],
    'compactness': [0.87, 0.84, 0.87, 0.83, 0.89, 0.87],
    'length': [5.2, 5.1, 5.2, 4.9, 5.5, 5.2],
    'width': [3.2, 3.1, 3.2, 2.9, 3.4, 3.2],
    'asymmetry': [2.2, 2.1, 2.2, 2.0, 2.3, 2.2],
    'groove_length': [5.6, 5.5, 5.6, 5.3, 5.8, 5.6],
}

df = pd.DataFrame(data)
# Mark duplicates as 1, unique as 0
df['duplicate'] = df.duplicated().astype(int)
X = df.drop('duplicate', axis=1)
y = df['duplicate']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
