import json
import math
from utils.text_processing import text_processing

# Load model once
with open('models/model_data.json') as f:
    loaded_data = json.load(f)

priors = loaded_data["priors"]
vocab = set(loaded_data["vocab"])
word_counts_table = loaded_data["word_counts_table"]

def predict(txt: str) -> dict:
    labels = ['positive', 'negative']
    tokens = text_processing(txt)

    log_probs = {}
    for label in labels:
        # Start with log prior
        log_prob = math.log(priors[label])
        
        # Add log likelihood for each token
        for token in tokens:
            # Laplace smoothing
            word_count = word_counts_table[label].get(token, 0)
            denom = len(vocab) + word_counts_table[f't_{label}']
            prob = (word_count + 1) / denom
            log_prob += math.log(prob)
        
        log_probs[label] = log_prob

    # Convert back from log-space for confidence normalization
    max_label = max(log_probs, key=log_probs.get)
    exp_probs = {label: math.exp(lp) for label, lp in log_probs.items()}
    total = sum(exp_probs.values())
    confidence = exp_probs[max_label] / total * 100

    return {
        'label': max_label,
        'confidence': f"{confidence:.2f}%"
    }
