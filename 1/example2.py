from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib  # for saving the model

# Load sample data
data = load_iris()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'iris_model_example2.pkl')

loaded_model = joblib.load("iris_model_example2.pkl")
print(loaded_model.predict([X_test[0]]))