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

## Encoding benchmark (nested CV)

### Neden yeni benchmark var?

Eski `car_evaluation_model.py` akışı bütün özelliklerde `LabelEncoder` kullanıyordu (artık one-hot `Pipeline` kullanır, aşağıya bakın). Bu yaklaşım `low=1`, `med=2`, `high=0` gibi alfabetik ve yapay sayısal ilişkiler üretebilir. Özellikle SVM, KNN ve Logistic Regression bu mesafeleri gerçekmiş gibi yorumlayabilir. Encoder'ın split öncesi bütün veri üzerinde fit edilmesi de pipeline dışı preprocessing oluşturur.

`encoding_benchmark.py` iki kontrollü yolu karşılaştırır:

| Encoding | Uygulama | Amaç |
| --- | --- | --- |
| Ordinal baseline | `ColumnTransformer` + `OrdinalEncoder(unknown_value=-1)` | Tek kolonlu sayısal baseline; legacy yaklaşımın kontrollü karşılığı |
| One-hot | `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")` | Nominal kategoriler arasında yapay mesafe oluşturmamak |

Her encoder yalnızca ilgili inner/outer fold'un eğitim bölümünde fit edilir.

### Modeller

- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- K-Nearest Neighbors
- Logistic Regression
- Gaussian Naive Bayes

Bu liste genişlik gösterisinden çok preprocessing etkisini farklı model ailelerinde görmek içindir.

### Çalıştırma

Tam nested CV:

```bash
python encoding_benchmark.py --outer-folds 5 --inner-folds 3 --seed 42
```

Hızlı smoke run:

```bash
python encoding_benchmark.py --outer-folds 3 --inner-folds 2 --quick
```

Script her model/encoding çifti için inner `GridSearchCV` ile macro F1 seçimi yapar ve bağımsız outer fold'da değerlendirir. Üretilen artifact'ler:

- `benchmark-results/nested_cv_folds.csv`
- `benchmark-results/summary.csv`
- `benchmark-results/results.json`
- `benchmark-results/README.md`

### Raporlanan metrikler

| Alan | Neden? |
| --- | --- |
| Accuracy | Legacy sonuçlarla kaba karşılaştırma |
| Balanced accuracy | Dengesiz sınıflarda her sınıfa eşit ağırlık |
| Macro F1 | Az görülen `good`/`vgood` sınıflarını saklamayan ana seçim metriği |
| Log loss | Olasılık kalitesi |
| Multiclass Brier | Olasılıkların one-hot hedefe karesel uzaklığı; düşük daha iyi |
| ECE | Confidence ile gerçekleşen doğruluk farkı; düşük daha iyi |
| Weighted OVR ROC AUC | Olasılık sıralama özeti |
| Confusion/error analysis | En sık gerçek→tahmin hata çiftleri ve sınıf bazlı precision/recall/F1 |

### Mevcut tekrarlanabilir sonuç

Bu PR'daki benchmark; `car.data` SHA-256 `2a9f028bd612f90ce7bb94dc798bd5748d6d1811ad4d740aad5c25187a4459a3`, seed 42, 5 outer fold ve 3 inner fold ile çalıştırıldı. Tam tablo, fold kayıtları ve seçilen hiperparametreler [`benchmark-results`](benchmark-results/README.md) altında bulunur.

Öne çıkan macro F1 sonuçları:

| Model | Ordinal | One-hot | One-hot farkı |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.3257 ± 0.0212 | 0.8863 ± 0.0252 | **+0.5606** |
| Naive Bayes | 0.3294 ± 0.0165 | 0.7394 ± 0.0378 | **+0.4100** |
| SVM | 0.9876 ± 0.0161 | 0.9978 ± 0.0049 | **+0.0102** |
| Gradient Boosting | 0.9644 ± 0.0162 | 0.9723 ± 0.0202 | **+0.0079** |
| Random Forest | 0.9633 ± 0.0107 | 0.9180 ± 0.0181 | -0.0453 |

Sonuç, “one-hot her zaman kazanır” demiyor. Yapay ordinal ilişkinin özellikle Logistic Regression ve Naive Bayes'i ciddi biçimde bozduğunu; bazı ağaç/mesafe tabanlı modellerde ise ordinal baseline'ın daha yüksek outer-fold skoru verdiğini gösteriyor.

### Sonuçların yorumu

Eski yüzde tablosu tek train/test split ve LabelEncoder akışından geliyordu; bu README bu sayıları güncel model kanıtı olarak tekrarlamaz. Kanonik karşılaştırma, komut çalıştırıldıktan sonra [`benchmark-results/README.md`](benchmark-results/README.md) ve ham JSON/CSV artifact'leridir.

One-hot'ın her modelde kazanması beklenmemelidir. Ağaç modelleri kontrollü ordinal baseline'daki bazı sıralamalardan yararlanabilir; lineer ve mesafe tabanlı modellerde ise yapay ordinal ilişkinin etkisi daha belirgin olabilir. Kararı model adı veya vocabulary büyüklüğü gibi sezgilerle değil, outer-fold dağılımıyla verin.


## Neden LabelEncoder kaldırıldı?

Girdi özellikleri nominal/karma kategorilerdir. Alfabetik `LabelEncoder` değerleri SVM, KNN, lojistik regresyon ve ağaç eşikleri için yapay bir sayısal sıra oluşturuyordu. Yeni pipeline, kategorileri one-hot olarak dönüştürür ve aynı preprocessing'i eğitim ile inference'ta birlikte saklar.

## Test

```bash
pytest -q
```

`tests/test_encoding_benchmark.py` encoding benchmark birimlerini, `tests/test_pipeline.py` eğitim/tahmin pipeline'ını kapsar.

## Veri

- Kaynak: [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- 1.728 kayıt
- 6 kategorik özellik
- 4 hedef sınıf: `unacc`, `acc`, `good`, `vgood`
