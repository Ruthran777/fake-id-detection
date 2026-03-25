from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        followers = int(request.form["followers"])
        following = int(request.form["following"])
        posts = int(request.form["posts"])
        bio = int(request.form["bio"])
        pic = int(request.form["pic"])
        verified = int(request.form["verified"])

        data = np.array([[followers, following, posts, bio, pic, verified]])

        result = model.predict(data)

        if result[0] == 1:
            prediction = "Fake Profile ❌"
        else:
            prediction = "Real Profile ✅"

        return render_template("index.html", prediction_text=prediction)

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)