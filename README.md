# Car Evaluation — classic ML benchmark

UCI Car Evaluation veri seti üzerinde kategorik encoding ve klasik sınıflandırma algoritmalarını karşılaştıran eğitim projesi. Veri seti 1.728 satır, 6 nominal/ordinal kategorik özellik ve 4 hedef sınıf içerir.

Bu repo “ileri ML” iddiası taşımaz. Küçük, iyi bilinen bir veri setinde preprocessing, nested cross-validation, calibration ve hata analizi öğrenmek için kullanılır.

## Neden yeni benchmark var?

Legacy `car_evaluation_model.py`, bütün özelliklerde `LabelEncoder` kullanır. Bu yaklaşım `low=1`, `med=2`, `high=0` gibi alfabetik ve yapay sayısal ilişkiler üretebilir. Özellikle SVM, KNN ve Logistic Regression bu mesafeleri gerçekmiş gibi yorumlayabilir. Encoder'ın split öncesi bütün veri üzerinde fit edilmesi de pipeline dışı preprocessing oluşturur.

`encoding_benchmark.py` iki kontrollü yolu karşılaştırır:

| Encoding | Uygulama | Amaç |
| --- | --- | --- |
| Ordinal baseline | `ColumnTransformer` + `OrdinalEncoder(unknown_value=-1)` | Tek kolonlu sayısal baseline; legacy yaklaşımın kontrollü karşılığı |
| One-hot | `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")` | Nominal kategoriler arasında yapay mesafe oluşturmamak |

Her encoder yalnızca ilgili inner/outer fold'un eğitim bölümünde fit edilir.

## Modeller

- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- K-Nearest Neighbors
- Logistic Regression
- Gaussian Naive Bayes

Bu liste genişlik gösterisinden çok preprocessing etkisini farklı model ailelerinde görmek içindir.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Encoding benchmark

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

## Raporlanan metrikler

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

## Mevcut tekrarlanabilir sonuç

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

## Sonuçların yorumu

Kökteki eski yüzde tablosu tek train/test split ve legacy encoding akışından geliyordu. Yeni README bu sayıları güncel model kanıtı olarak tekrarlamaz. Kanonik karşılaştırma, komut çalıştırıldıktan sonra [`benchmark-results/README.md`](benchmark-results/README.md) ve ham JSON/CSV artifact'leridir.

One-hot'ın her modelde kazanması beklenmemelidir. Ağaç modelleri kontrollü ordinal baseline'daki bazı sıralamalardan yararlanabilir; lineer ve mesafe tabanlı modellerde ise yapay ordinal ilişkinin etkisi daha belirgin olabilir. Kararı model adı veya vocabulary büyüklüğü gibi sezgilerle değil, outer-fold dağılımıyla verin.

## Legacy uygulamalar

| Dosya | Durum |
| --- | --- |
| `car_evaluation_model.py` | Legacy eğitim/görselleştirme akışı; LabelEncoder kullanır |
| `interactive_predictor.py` / `web_app.py` | Legacy model/encoder artifact'leriyle tahmin arayüzleri |
| `encoding_benchmark.py` | Kanonik encoding karşılaştırması ve değerlendirme |
| `advanced_model_visualizer.py` | Görselleştirme deneyi |

Benchmark sonucu seçilmeden legacy predictor'ı sessizce yeni pipeline'a geçirmek doğru değildir. Yeni üretim artifact'i istenecekse seçilen pipeline tam olarak fit edilmeli, encoder+model birlikte kaydedilmeli ve UI entegrasyonu ayrıca test edilmelidir.

## Test

```bash
python -m unittest discover -s tests -v
```

## Veri ve sınırlamalar

- Kaynak: [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- Veri deterministik karar kurallarından türetilmiştir; gerçek araç satın alma davranışını temsil etmez.
- Küçük veri setindeki çok yüksek doğruluk gerçek dünyaya genellenemez.
- Aynı veri üzerinde çok sayıda model denemek araştırmacı serbestliği yaratır; nested CV bunu azaltır ama harici doğrulamanın yerini tutmaz.
- Calibration metriği, deploy edilmiş olasılıkların zaman içinde kalibre kalacağını garanti etmez.
