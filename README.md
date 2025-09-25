# Sentiment Analysis API

This project provides a simple sentiment analysis pipeline using a Naive Bayes classifier trained on the Amazon Reviews dataset.  
It exposes a Flask API to make predictions on text input.

## Dataset
The model is trained using the Amazon Reviews dataset:  
[Amazon Reviews Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data)

---

## Folder Structure

```
.
├── data
│   └── test.ft.txt                 # Sample dataset file
├── models
│   └── model_data.json             # Trained model data (priors, vocabulary, word counts)
├── src
│   ├── predict.py                  # Prediction logic using trained model
│   ├── train.py                    # Script to train and generate model_data.json
│   └── __init__.py
├── utils
│   ├── text_processing.py          # Tokenization and text preprocessing functions
│   └── __init__.py
├── app.py                          # Flask API to expose sentiment analysis predictions
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## Setup

1. Clone the repository and navigate into the project folder.

```bash
git clone <repo_url>
cd <repo_name>
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate    # On Windows
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

---

## Training

Run the training script to generate the `model_data.json` file inside the `models` directory.

```bash
python src/train.py
```

---

## Prediction via Flask API

1. Start the Flask app:

```bash
python app.py
```

2. Send a POST request with JSON input:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "This product is amazing!"}'
```

3. Example Response:

```json
{
  "label": "positive",
  "confidence": "95.34%"
}
```

---

## File Notes

- `data/` contains raw input files for training.
- `models/` contains the trained model data (`model_data.json`).
- `src/` holds training and prediction logic.
- `utils/` contains helper functions like text preprocessing.
- `app.py` is the main Flask application.

---

## Requirements

Dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## License

This project is for educational purposes and not intended for production use.
