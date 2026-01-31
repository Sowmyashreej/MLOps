import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the Wine dataset (Modification 1)
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Use Random Forest (Modification 2)
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

# 3. Generate a Report
predictions = model.predict(X_test)
report = classification_report(y_test, predictions)

print("--- Model Performance Report ---")
print(report)

# 4. Save metrics to a file (Modification 3)
with open("results.txt", "w") as f:
    f.write(report)

