import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the training and testing data
X_train = np.load(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\test_X.npy")
y_train = np.load(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\test_y.npy")
X_test = np.load(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\train_X.npy")
y_test = np.load(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\train_y.npy")


model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to disk
joblib.dump(model, r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\model\speaker_model.pkl")
print("Model saved as 'models/speaker_model.pkl'")
