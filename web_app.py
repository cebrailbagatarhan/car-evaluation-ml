"""
Web Tabanlı Car Evaluation Predictor
====================================

Flask kullanarak web tabanlı veri girişi sistemi
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os

# Predictor sınıfını import et
sys.path.append(os.path.dirname(__file__))
from interactive_predictor import CarEvaluationPredictor, predict_car

app = Flask(__name__)

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin API endpoint'i"""
    try:
        data = request.json
        
        # Gerekli alanları kontrol et
        required_fields = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Eksik alan: {field}'}), 400
        
        # Tahmin yap
        prediction, probabilities = predict_car(
            data['buying'], data['maint'], data['doors'],
            data['persons'], data['lug_boot'], data['safety']
        )
        
        if prediction is None:
            return jsonify({'error': 'Model yüklenemedi'}), 500
        
        # Sonucu döndür
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Toplu tahmin API endpoint'i"""
    try:
        data = request.json
        
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({'error': 'Geçersiz veri formatı'}), 400
        
        predictor = CarEvaluationPredictor()
        if not predictor.load_model():
            return jsonify({'error': 'Model yüklenemedi'}), 500
        
        results = []
        for item in data['data']:
            try:
                prediction, probabilities = predictor.predict_single(item)
                results.append({
                    'input': item,
                    'prediction': prediction,
                    'probabilities': probabilities
                })
            except Exception as e:
                results.append({
                    'input': item,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)