# app.py
from flask import Flask, render_template, request, jsonify
from generate_headline import generate_headline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    headline = generate_headline(text)
    return jsonify({"headline": headline})

if __name__ == "__main__":
    app.run(debug=True)
