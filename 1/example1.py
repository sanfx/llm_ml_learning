# 1. Import dependencies
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # for saving model

# 2. Load dataset
X, y = load_iris(return_X_y=True)

# 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save model to file
joblib.dump(model, "iris_model_example1.pkl")

# Later in another script:
loaded_model = joblib.load("iris_model_example1.pkl")
print(loaded_model.predict([X_test[0]]))
