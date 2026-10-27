"""Flask tabanlı Car Evaluation API ve web arayüzü."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from interactive_predictor import CarEvaluationPredictor


app = Flask(__name__)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        predictor = CarEvaluationPredictor()
        if not predictor.load_model():
            raise RuntimeError(predictor.last_error or "Model yüklenemedi")
        _predictor = predictor
    return _predictor


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    try:
        get_predictor()
    except RuntimeError:
        return jsonify({"status": "not_ready"}), 503
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object bekleniyor"}), 400

    try:
        prediction, probabilities = get_predictor().predict_single(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError:
        app.logger.exception("Model artifact yüklenemedi")
        return jsonify({"error": "Model hazır değil"}), 503
    except Exception:
        app.logger.exception("Tahmin sırasında beklenmeyen hata")
        return jsonify({"error": "Tahmin başarısız"}), 500

    return jsonify(
        {
            "prediction": prediction,
            "probabilities": probabilities,
            "input_data": data,
        }
    )


@app.post("/batch_predict")
def batch_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return jsonify({"error": "'data' alanı bir liste olmalıdır"}), 400
    if len(payload["data"]) > 100:
        return jsonify({"error": "Bir istekte en fazla 100 kayıt desteklenir"}), 400

    try:
        predictor = get_predictor()
    except RuntimeError:
        app.logger.exception("Model artifact yüklenemedi")
        return jsonify({"error": "Model hazır değil"}), 503

    results = []
    for item in payload["data"]:
        try:
            prediction, probabilities = predictor.predict_single(item)
            results.append(
                {
                    "input": item,
                    "prediction": prediction,
                    "probabilities": probabilities,
                }
            )
        except ValueError as exc:
            results.append({"input": item, "error": str(exc)})
        except Exception:
            app.logger.exception("Batch tahmin sırasında beklenmeyen hata")
            results.append({"input": item, "error": "Tahmin başarısız"})

    return jsonify({"results": results})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
