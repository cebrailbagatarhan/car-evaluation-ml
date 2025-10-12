# Car Evaluation Predictor - Gereksinimler

## Veri Girişi Yöntemleri

Bu proje, Car Evaluation modeli için çeşitli veri giriş yöntemleri sunar:

### 1. İnteraktif Konsol Uygulaması

**Dosya**: `interactive_predictor.py`

```bash
python interactive_predictor.py
```

**Özellikler**:
- Menü tabanlı interaktif sistem
- Adım adım veri girişi
- Tek tahmin ve toplu tahmin desteği
- Görsel sonuç gösterimi
- Örnek dosya oluşturma

### 2. Web Tabanlı Arayüz

**Dosyalar**: `web_app.py`, `templates/index.html`

```bash
# Flask'ı yükle
pip install flask

# Web uygulamasını başlat
python web_app.py
```

**Özellikler**:
- Modern web arayüzü
- Dropdown menülerle kolay seçim
- Anlık tahmin sonuçları
- Olasılık grafikleri
- Responsive tasarım

### 3. Komut Satırı Arayüzü

**Dosya**: `cli_predictor.py`

```bash
python cli_predictor.py --buying high --maint med --doors 4 --persons 4 --lug_boot big --safety high
```

**Özellikler**:
- Tek komutla tahmin
- JSON veya text çıktı
- Script entegrasyonu için ideal
- Batch işlemler için uygun

### 4. Dosyadan Toplu Veri Girişi

**CSV Formatı**: `sample_batch_input.csv`
```csv
buying,maint,doors,persons,lug_boot,safety
high,med,4,4,big,high
low,low,4,more,big,high
```

**JSON Formatı**: `sample_batch_input.json`
```json
[
  {
    "buying": "high",
    "maint": "med",
    "doors": "4",
    "persons": "4", 
    "lug_boot": "big",
    "safety": "high"
  }
]
```

### 5. Programmatic API

```python
from interactive_predictor import predict_car

# Direkt fonksiyon çağrısı
prediction, probabilities = predict_car(
    buying='high',
    maint='med', 
    doors='4',
    persons='4',
    lug_boot='big',
    safety='high'
)

print(f"Tahmin: {prediction}")
```

## Geçerli Değerler

### Özellikler ve Değerleri:

- **buying** (Satın alma fiyatı): `vhigh`, `high`, `med`, `low`
- **maint** (Bakım maliyeti): `vhigh`, `high`, `med`, `low`
- **doors** (Kapı sayısı): `2`, `3`, `4`, `5more`
- **persons** (Kişi kapasitesi): `2`, `4`, `more`
- **lug_boot** (Bagaj boyutu): `small`, `med`, `big`
- **safety** (Güvenlik seviyesi): `low`, `med`, `high`

### Tahmin Sınıfları:

- **unacc**: Kabul edilemez
- **acc**: Kabul edilebilir
- **good**: İyi
- **vgood**: Çok iyi

## Kurulum ve Kullanım

### 1. Gerekli Paketleri Yükle
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install flask  # Web arayüzü için
```

### 2. Modeli Eğit (İlk çalıştırmada)
```bash
python interactive_predictor.py
# Menüden seçeneği takip edin
```

### 3. Veri Girişi Yöntemini Seçin

#### İnteraktif Konsol:
```bash
python interactive_predictor.py
```

#### Web Arayüzü:
```bash
python web_app.py
# Tarayıcıda http://localhost:5000 açın
```

#### Komut Satırı:
```bash
python cli_predictor.py --buying high --maint med --doors 4 --persons 4 --lug_boot big --safety high --format json
```

#### Toplu İşlem:
```bash
python interactive_predictor.py
# Menüden "Dosyadan toplu tahmin" seçin
# sample_batch_input.csv veya sample_batch_input.json kullanın
```

## Dosya Yapısı

```
car+evaluation/
├── car.data                    # Orijinal veri seti
├── interactive_predictor.py    # Ana interaktif sistem
├── web_app.py                  # Flask web uygulaması
├── cli_predictor.py            # Komut satırı arayüzü
├── sample_batch_input.csv      # Örnek CSV dosyası
├── sample_batch_input.json     # Örnek JSON dosyası
├── templates/
│   └── index.html              # Web arayüzü
├── car_evaluation_model.pkl    # Eğitilmiş model (otomatik oluşur)
├── label_encoders.pkl          # Encoder'lar (otomatik oluşur)
└── encoders.json               # JSON format encoder'lar
```

## Örnek Kullanımlar

### 1. Hızlı Test
```python
from interactive_predictor import predict_car

result, probs = predict_car('high', 'med', '4', '4', 'big', 'high')
print(f"Sonuç: {result}")
```

### 2. Web API Çağrısı
```javascript
fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        buying: 'high',
        maint: 'med',
        doors: '4',
        persons: '4',
        lug_boot: 'big',
        safety: 'high'
    })
})
```

### 3. Batch Processing
```bash
python interactive_predictor.py
# Menü: 3 - Dosyadan toplu tahmin
# Dosya: sample_batch_input.csv
```

Bu sistem, farklı kullanım senaryolarına uygun esnek veri giriş seçenekleri sunar!