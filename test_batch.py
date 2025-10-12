"""
Test scripti - Toplu tahmin
"""
from interactive_predictor import CarEvaluationPredictor

# Predictor oluştur
p = CarEvaluationPredictor()
p.load_model()

# CSV'den toplu tahmin
print("📊 CSV'den Toplu Tahmin Testi:")
print("=" * 40)

try:
    results = p.batch_input_from_file('sample_input.csv')
    if results:
        for i, r in enumerate(results, 1):
            print(f"{i}. Tahmin: {r['prediction'].upper()}")
    else:
        print("❌ CSV dosyasından veri okunamadı")
except Exception as e:
    print(f"❌ Hata: {e}")

# JSON'dan toplu tahmin  
print("\n📊 JSON'dan Toplu Tahmin Testi:")
print("=" * 40)

try:
    results = p.batch_input_from_file('sample_input.json')
    if results:
        for i, r in enumerate(results, 1):
            print(f"{i}. Tahmin: {r['prediction'].upper()}")
    else:
        print("❌ JSON dosyasından veri okunamadı")
except Exception as e:
    print(f"❌ Hata: {e}")