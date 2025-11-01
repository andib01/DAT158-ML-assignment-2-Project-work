from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/mushroom_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
cat_info = joblib.load("models/mushroom_categories.pkl")

columns = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color",
    "ring-number", "ring-type",
    "spore-print-color", "population", "habitat"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        user_input = {col: request.form[col] for col in columns}
        X = pd.DataFrame([user_input])

        for col in columns:
            X[col] = pd.Categorical(X[col], categories=cat_info[col])

        y_pred = model.predict(X)[0]
        label = encoder.inverse_transform([int(y_pred)])[0]

    if label == "e":
        prediction = "This mushroom is edible!"
    else:
        prediction = "This mushroom is poisonous!"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)