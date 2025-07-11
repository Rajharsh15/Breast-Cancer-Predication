import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_prepare_data():
    # Load dataset
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    
    # Create DataFrame
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    
    # Add target column
    data_frame['diagnosis'] = breast_cancer_dataset.target
    
    return data_frame

def train_model(data_frame):
    # Separate features and target
    X = data_frame.drop(columns='diagnosis', axis=1)
    Y = data_frame['diagnosis']
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    
    # Initialize and train model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
    
    # Evaluate accuracy
    train_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))
    
    return model, train_accuracy, test_accuracy, list(X.columns)

def predict_cancer(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 0:
        return "Malignant ⚠️"
    else:
        return "Benign ✅"

# Example usage
if __name__ == "__main__":
    df = load_and_prepare_data()
    
    model, train_acc, test_acc, feature_names = train_model(df)
    
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    
    """# Example input (30 features)
    input_example = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,
                     0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,
                     0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,
                     15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,
                     0.2977,0.07259)
    
    result = predict_cancer(model, input_example)
    print("Prediction:", result)"""
