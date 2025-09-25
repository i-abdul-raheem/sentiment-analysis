from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route("/", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Request must contain 'text' field"}), 400

    result = predict(data["text"])
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)