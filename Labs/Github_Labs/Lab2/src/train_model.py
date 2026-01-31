import os
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Wine dataset (Modification for Lab Requirement)
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2. Train Random Forest (High accuracy to pass threshold)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. FIX: Save model to the specific 'models' folder Lab 2 expects
# This path moves up from 'src' to the 'Lab2' root, then into 'models'
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '..', 'models', 'latest_model.pkl')

# Ensure directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the model
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

