import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Extract data from form
    name = request.form["name"]
    platform = request.form["platform"]
    r_date = request.form["r-date"]
    user_score = request.form["user-score"]
    developer = request.form["developer"]
    genre = request.form["genre"]
    critics = int(request.form["critics"])
    users = int(request.form["users"])

    # Create a DataFrame for model input
    input_data = pd.DataFrame([[name, platform, r_date, user_score, developer, genre, critics, users]], 
                                columns=['name', 'platform', 'r-date', 'user score', 'developer', 'genre', 'critics', 'users'])

    # Predict the score
    prediction = model.predict(input_data)[0]

    # Return the result
    return render_template("index.html", prediction_text="Predicted MetaCritic Score: {:.2f}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)