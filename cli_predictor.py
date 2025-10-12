"""
Komut Satırı Veri Girişi Sistemi
================================

Bu script, komut satırından argümanlar alarak tahmin yapar.
"""

import argparse
import json
import sys
from interactive_predictor import predict_car

def main():
    parser = argparse.ArgumentParser(description='Car Evaluation Predictor - Komut Satırı Arayüzü')
    
    # Parametreleri tanımla
    parser.add_argument('--buying', choices=['vhigh', 'high', 'med', 'low'], 
                       required=True, help='Satın alma fiyatı')
    parser.add_argument('--maint', choices=['vhigh', 'high', 'med', 'low'], 
                       required=True, help='Bakım maliyeti')
    parser.add_argument('--doors', choices=['2', '3', '4', '5more'], 
                       required=True, help='Kapı sayısı')
    parser.add_argument('--persons', choices=['2', '4', 'more'], 
                       required=True, help='Kişi kapasitesi')
    parser.add_argument('--lug_boot', choices=['small', 'med', 'big'], 
                       required=True, help='Bagaj boyutu')
    parser.add_argument('--safety', choices=['low', 'med', 'high'], 
                       required=True, help='Güvenlik seviyesi')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Çıktı formatı (varsayılan: text)')
    
    args = parser.parse_args()
    
    # Tahmin yap
    prediction, probabilities = predict_car(
        args.buying, args.maint, args.doors,
        args.persons, args.lug_boot, args.safety
    )
    
    if prediction is None:
        print("❌ Model yüklenemedi. Önce modeli eğitin.", file=sys.stderr)
        sys.exit(1)
    
    # Sonucu formatla
    if args.format == 'json':
        result = {
            'input': vars(args),
            'prediction': prediction,
            'probabilities': probabilities
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Text format
        class_descriptions = {
            'unacc': 'Kabul Edilemez',
            'acc': 'Kabul Edilebilir', 
            'good': 'İyi',
            'vgood': 'Çok İyi'
        }
        
        print("🚗 ARABA DEĞERLENDİRME SONUCU")
        print("=" * 40)
        print(f"🎯 Tahmin: {prediction.upper()}")
        print(f"📝 Açıklama: {class_descriptions[prediction]}")
        
        if probabilities:
            print("\n📊 Olasılıklar:")
            for class_name, prob in probabilities.items():
                print(f"  {class_name:>6}: %{prob*100:5.1f}")

if __name__ == "__main__":
    main()