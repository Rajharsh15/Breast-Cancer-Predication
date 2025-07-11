from flask import Flask, render_template, request
from model_code import load_and_prepare_data, train_model, predict_cancer

app = Flask(__name__)

# Load and prepare data first
data_frame = load_and_prepare_data()

# Then train the model and get features
model, train_accuracy , test_accuracy,features = train_model(data_frame)

@app.route('/')
def home():
    return render_template("index.html", features=features,train_accuracy=train_accuracy, test_accuracy= test_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        result = predict_cancer(model, input_data)
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result, features=features, input_values=request.form, train_accuracy = train_accuracy, test_accuracy = test_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
