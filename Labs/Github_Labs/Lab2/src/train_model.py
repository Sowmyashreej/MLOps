import os
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

script_dir = os.path.dirname(__file__)

model_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'latest_model.pkl')

joblib.dump(model, model_path)
print(f"Success: Model saved at {model_path}")
