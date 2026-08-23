# Car Evaluation ML

UCI Car Evaluation veri seti için kategorik preprocessing, model seçimi, CLI ve Flask API örneği.

Bu proje eğitim amaçlıdır. README'de sabit bir doğruluk iddiası tutulmaz; her koşunun sonuçları `models/car_evaluation_pipeline.metadata.json` dosyasına yazılır.

## Kurulum

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Modeli eğit

```bash
python car_evaluation_model.py
```

Eğitim akışı:

1. Veriyi stratified train/test olarak ayırır.
2. Kategorileri `OneHotEncoder` ile sklearn `Pipeline` içinde dönüştürür.
3. Yedi aday modeli yalnızca training split üzerindeki 5-fold CV ile karşılaştırır.
4. Seçilen modeli training split üzerinde fit eder.
5. Hold-out test setini yalnızca bir kez raporlar.
6. Model ve preprocessing'i tek artifact olarak kaydeder.

Çıktılar:

- `models/car_evaluation_pipeline.joblib`
- `models/car_evaluation_pipeline.metadata.json`

Joblib/pickle tabanlı artifact'ler kod çalıştırabilir; yalnızca bu eğitim scriptinin ürettiği güvenilen yerel artifact'i yükleyin.

## Tahmin

İnteraktif:

```bash
python interactive_predictor.py
```

Komut satırı:

```bash
python cli_predictor.py \
  --buying low \
  --maint med \
  --doors 4 \
  --persons more \
  --lug_boot big \
  --safety high \
  --format json
```

## Web arayüzü ve API

```bash
python web_app.py
```

Varsayılan adres: <http://127.0.0.1:5000>

`FLASK_DEBUG` varsayılan olarak kapalıdır. Dinlenecek adres ve port gerektiğinde ortam değişkenleriyle ayarlanır:

```bash
FLASK_HOST=0.0.0.0 PORT=5000 python web_app.py
```

Endpoint'ler:

- `GET /health`
- `POST /predict`
- `POST /batch_predict` — istek başına en fazla 100 kayıt

Örnek:

```json
{
  "buying": "low",
  "maint": "med",
  "doors": "4",
  "persons": "more",
  "lug_boot": "big",
  "safety": "high"
}
```

## Neden LabelEncoder kaldırıldı?

Girdi özellikleri nominal/karma kategorilerdir. Alfabetik `LabelEncoder` değerleri SVM, KNN, lojistik regresyon ve ağaç eşikleri için yapay bir sayısal sıra oluşturuyordu. Yeni pipeline, kategorileri one-hot olarak dönüştürür ve aynı preprocessing'i eğitim ile inference'ta birlikte saklar.

## Test

```bash
pytest -q
```

## Veri

- Kaynak: [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- 1.728 kayıt
- 6 kategorik özellik
- 4 hedef sınıf: `unacc`, `acc`, `good`, `vgood`
