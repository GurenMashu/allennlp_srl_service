import logging
import warnings
from pathlib import Path
from flask import Flask, jsonify, request
from waitress import serve

warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used.*")

from allennlp.predictors.predictor import Predictor

MODEL_PATH = Path(__file__).parent / "models" / "structured-prediction-srl-bert.2020.12.15.tar.gz"
PORT = 5000
HOST = "0.0.0.0"

logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
predictor = None

def load_model():
    global predictor
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH.absolute()}")
    
    logger.info(f"Loading SRL model from {MODEL_PATH}")
    predictor = Predictor.from_path(MODEL_PATH, cuda_device = -1)
    logger.info("SRL Model loaded successfully")

@app.route("/health", methods = ["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})

@app.route("/predict", methods = ["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "model_not_loaded"}), 504
    
    try:
        data = request.get_json()

        if not data or "sentence" not in data:
            return jsonify({"error": "Missing 'sentence' field"}), 400
        
        sentence = data["sentence"]

        if not sentence or not isinstance(sentence, str):
            return jsonify({"error": "Sentence must be non-empty string"}), 400
        
        result = predictor.predict(sentence = sentence)
    
        return jsonify(result)

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": str(e)}), 503
    
@app.route("/batch_predict", methods = ["POST"])
def batch_predict():
    if predictor is None:
        return jsonify({"error": "model_not_loaded"}), 504
    
    try:
        data = request.get_json()

        if not data or "sentence" not in data:
            return jsonify({"error": "Missing 'sentence' field"}), 400
        
        sentences = data["sentence"]

        if not sentences or not isinstance(sentence, list):
            return jsonify({"error": "Sentence must be non-empty list"}), 400
        
        result = predictor.predict_json({"sentence": sentence})
    
        results = []
        for sentence in sentences:
            if not isinstance(sentence, str):
               return jsonify({"error": "Sentence must be string"}), 400
            else:
                result = predictor.predict_json({"sentence": sentence})
                results.append(result)
        
        return jsonify({"results": results})
            
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": str(e)}), 503


if __name__ == "__main__":
    load_model()
    
    logger.info(f"Starting SRL Server on http://{HOST}:{PORT}")
    logger.info(f"Endpoints: ")
    logger.info(f"  - Health: http://{HOST}:{PORT}/health")
    logger.info(f"  - Predict: http://{HOST}:{PORT}/predict")
    logger.info(f"  - Batch Predict: https://{HOST}:{PORT}/batch_predict")

    serve(app, host = HOST, port = PORT, threads = 4)