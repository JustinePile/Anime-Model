# Required imports
from flask import Flask, request, jsonify
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create the Flask app
app = Flask(__name__)

# Read the CSV file into a pandas DataFrame
file_path = 'data/MAL-anime.csv'
df = pd.read_csv(file_path)

# Prepare the data for modeling
X = df.drop(columns=['Score'])  # Features (all columns except 'Score')
y = df['Score']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create and train the linear regression model

def train_linear_regression_model(X_train_encoded, y_train):
    model = LinearRegression()
    model.fit(X_train_encoded, y_train)
    return model

# Combine the training and test sets to get all possible categorical values
combined_data = pd.concat([X, X_test], axis=0)

# Perform one-hot encoding on categorical columns with all possible values
X_encoded = pd.get_dummies(combined_data)

# Split the data into training and testing sets again
X_train_encoded = X_encoded.iloc[:len(X_train)]
X_test_encoded = X_encoded.iloc[len(X_train):]

# Create and train the linear regression model using the function
model = train_linear_regression_model(X_train_encoded, y_train)

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Perform one-hot encoding on the input data
        X_input = pd.DataFrame(data)  # Convert JSON to DataFrame
        X_input_encoded = pd.get_dummies(X_input)

        # Make predictions using the trained model
        y_pred = model.predict(X_input_encoded)

        # Return the predictions as JSON
        return jsonify({'predictions': y_pred.tolist()})

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500

# Run the Flask app on localhost:5000
if __name__ == '__main__':
    app.run(debug=True)
