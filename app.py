import pickle
import numpy as np
from os import path
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/result", methods=["POST"])
def result():
    # Example inputs: replace with your actual feature names
    country = str(request.form["Country"])
    continent = str(request.form["Continent"])
    Beer_Servings = float(request.form["Beer_Servings"])
    Wine_Servings = float(request.form["Wine_Servings"])
    Spirit_Servings = float(request.form["Spirit_Servings"])

    model_path = path.join(app.root_path, "static", "lr_reg.pkl")
    with open(model_path, "rb") as f:
        lr_model = pickle.load(f)

    features = np.array([[Beer_Servings, Wine_Servings, Spirit_Servings]])
    prediction = lr_model.predict(features)[0]

    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(port=8000)
