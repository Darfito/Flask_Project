import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[x]) for x in request.form]
        prediction = model.predict([features])
        if prediction == 0:
            return render_template("index.html", prediction_text="tidak berpengaruh", prediction=prediction)
        elif prediction == 1:
            return render_template("index.html", prediction_text="berpengaruh", prediction=prediction)
    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
    