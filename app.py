from flask import Flask, render_template, request, jsonify
import random
import nltk
from chatbot import get_response  

nltk.download('punkt')
nltk.download('wordnet')


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form["user_input"]
    response = get_response(user_input)  
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
