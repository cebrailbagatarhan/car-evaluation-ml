# Car Evaluation Machine Learning Project

🚗 **Araba Değerlendirme Makine Öğrenmesi Projesi**

UCI Car Evaluation veri setini kullanarak 7 farklı makine öğrenmesi algoritması ile otomobil kabul edilebilirlik tahmini yapan kapsamlı bir proje.

## 🎯 Proje Özellikleri

### 📊 Veri Seti
- **Kaynak:** UCI Machine Learning Repository
- **Örnekler:** 1,728 araba
- **Özellikler:** 6 kategorik özellik
- **Sınıflar:** 4 kabul edilebilirlik seviyesi (unacc, acc, good, vgood)

### 🤖 Makine Öğrenmesi Modelleri
1. **Decision Tree** - %98.55 doğruluk ⭐
2. **Random Forest** - %98.27 doğruluk
3. **Gradient Boosting** - %97.98 doğruluk
4. **SVM** - %92.77 doğruluk
5. **K-Nearest Neighbors** - %90.75 doğruluk
6. **Logistic Regression** - %69.08 doğruluk
7. **Naive Bayes** - %64.16 doğruluk

## 🚀 Özellikler

### 📈 Gelişmiş Görselleştirmeler
- ✅ ROC Curves ve AUC skorları
- ✅ Confusion Matrix analizi
- ✅ Learning Curves
- ✅ Feature Importance analizi
- ✅ 3D interaktif görselleştirmeler (Plotly)
- ✅ Decision Tree yapı görselleştirmeleri (3 seviye detay)
- ✅ Model performans karşılaştırmaları

### 🎮 İnteraktif Tahmin Sistemi
- 🖥️ **Komut satırı menüsü** - 7 farklı seçenek
- 🌐 **Web arayüzü** - Flask tabanlı
- 📝 **5 farklı veri giriş yöntemi:**
  1. İnteraktif adım adım girdi
  2. Manuel tek seferde girdi
  3. Dosyadan toplu tahmin
  4. Örnek dosya oluşturma
  5. Model görselleştirme

### 🌳 Karar Ağacı Derinliği
- **TAM DETAY:** 14 seviye, 169 düğüm, 254 kural satırı
- **ORTA DETAY:** İlk 6 seviye (dengeli görüntü)
- **ÖZET:** İlk 3 seviye (hızlı bakış)

## 📦 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

**Gerekli kütüphaneler:**
- Python 3.13+
- pandas 2.2.3
- numpy 2.2.3
- scikit-learn 1.6.1
- matplotlib 3.10.3
- seaborn 0.13.2
- plotly (3D görselleştirmeler için)
- flask (web arayüzü için)

## 🎯 Kullanım

### 1. Ana Model Eğitimi ve Analizi
```bash
python car_evaluation_model.py
```
- 7 model eğitir ve karşılaştırır
- Görselleştirmeler oluşturur
- Model dosyalarını kaydeder

### 2. İnteraktif Tahmin Sistemi
```bash
python interactive_predictor.py
```
**Menü seçenekleri:**
1. İnteraktif veri girişi
2. Manuel veri girişi
3. Dosyadan toplu tahmin
4. Örnek dosyalar oluştur
5. Model görselleştirme
6. Gelişmiş model analizi
7. Çıkış

### 3. Model Görselleştirme
```bash
python model_visualizer.py
```
- Karar ağacı yapısı (3 farklı detay seviyesi)
- Özellik önemleri
- Ağaç derinlik analizi
- Model mimarisi diyagramları

### 4. Gelişmiş Analiz
```bash
python advanced_model_visualizer.py
```
- 9 farklı görselleştirme türü
- ROC curves
- Learning curves
- Confusion matrices
- 3D feature space

### 5. Web Arayüzü
```bash
python web_app.py
```
Tarayıcıda açın: `http://localhost:5000`

## 📁 Proje Yapısı

```
car-evaluation/
├── car.data                          # Veri seti
├── car_evaluation_model.py           # Ana model eğitimi
├── interactive_predictor.py          # İnteraktif tahmin sistemi
├── model_visualizer.py               # Karar ağacı görselleştirme
├── advanced_model_visualizer.py      # Gelişmiş analiz
├── web_app.py                        # Flask web uygulaması
├── requirements.txt                  # Gerekli kütüphaneler
├── models/                           # Eğitilmiş modeller
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── label_encoders.pkl
├── templates/                        # HTML şablonları
│   └── index.html
└── outputs/                          # Görselleştirmeler
    ├── decision_tree_FULL_DETAIL.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── ...
```

## 📊 Model Performansları

| Model | Test Doğruluğu | CV Ortalama | CV Std |
|-------|---------------|-------------|--------|
| Decision Tree | **98.55%** | 97.10% | 1.41% |
| Random Forest | 98.27% | 96.31% | 1.29% |
| Gradient Boosting | 97.98% | 97.97% | 0.93% |
| SVM | 92.77% | 88.71% | 1.00% |
| K-Nearest Neighbors | 90.75% | 88.28% | 1.47% |
| Logistic Regression | 69.08% | 69.54% | 0.65% |
| Naive Bayes | 64.16% | 64.62% | 2.71% |

## 🎨 Görselleştirme Örnekleri

### Karar Ağacı Detay Seviyeleri
- **TAM DETAY** (40x30 inch): 14 seviye, 169 düğüm
- **ORTA DETAY** (30x20 inch): İlk 6 seviye
- **ÖZET** (20x15 inch): İlk 3 seviye

### Oluşturulan Dosyalar
- 📊 9 farklı PNG görselleştirme
- 📈 2 HTML 3D interaktif grafik
- 📝 3 detaylı metin raporu
- 🌳 Karar ağacı SVG/PDF (opsiyonel)

## 🔍 Özellikler

### Giriş Özellikleri
1. **buying** - Satın alma fiyatı (vhigh, high, med, low)
2. **maint** - Bakım maliyeti (vhigh, high, med, low)
3. **doors** - Kapı sayısı (2, 3, 4, 5more)
4. **persons** - Kişi kapasitesi (2, 4, more)
5. **lug_boot** - Bagaj boyutu (small, med, big)
6. **safety** - Güvenlik (low, med, high)

### Hedef Sınıflar
- **unacc** - Kabul edilemez (70%)
- **acc** - Kabul edilebilir (22%)
- **good** - İyi (4%)
- **vgood** - Çok iyi (4%)

## 🎓 Teknik Detaylar

### Preprocessing
- Label Encoding (tüm kategorik değişkenler)
- Train-test split (80-20)
- Cross-validation (5-fold)

### Hiperparametre Optimizasyonu
- GridSearchCV ile Decision Tree optimizasyonu
- En iyi parametreler otomatik seçilir

### Font Desteği
- Türkçe karakter desteği
- Güvenli font fallback sistemi
- Cross-platform uyumlu

## 📈 Örnek Kullanım

### Tek Tahmin
```python
from car_evaluation_model import CarEvaluationModel

model = CarEvaluationModel('car.data')
model.load_data()
model.prepare_data()
model.train_all_models()

prediction = model.predict_single({
    'buying': 'low',
    'maint': 'med', 
    'doors': '4',
    'persons': 'more',
    'lug_boot': 'big',
    'safety': 'high'
})
print(prediction)  # 'vgood'
```

### Toplu Tahmin
```python
import pandas as pd

df = pd.read_csv('sample_batch_input.csv')
predictions = model.predict_batch(df)
```

## 🛠️ Geliştirme

### Test Ortamı
- Python 3.13.3
- Windows/Linux/Mac uyumlu
- VS Code ile geliştirildi

### Kod Kalitesi
- PEP 8 uyumlu
- Kategorize import'lar
- Detaylı dokümantasyon
- Türkçe ve İngilizce destekli

## 📝 Lisans

Bu proje eğitim amaçlıdır ve UCI Machine Learning Repository'den alınan açık kaynak veri setini kullanmaktadır.

## 👤 Yazar

**Cebrail Bağatarhan**
- GitHub: [@cebrailbagatarhan](https://github.com/cebrailbagatarhan)

## 🙏 Teşekkürler

- UCI Machine Learning Repository
- Scikit-learn kütüphanesi
- Matplotlib, Seaborn, Plotly görselleştirme araçları

## 📚 Kaynaklar

- [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

**Son Güncelleme:** Ekim 2025
