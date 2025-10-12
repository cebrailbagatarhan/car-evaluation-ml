# Car Evaluation Dataset - Machine Learning Analizi

## Proje Özeti

UCI Car Evaluation veri seti üzerinde kapsamlı bir makine öğrenmesi analizi gerçekleştirildi. Bu analiz, otomobillerin kabul edilebilirliğini çeşitli faktörlere göre değerlendiren bir sınıflandırma problemi üzerine odaklanmaktadır.

## Veri Seti Hakkında

**Kaynak**: UCI Machine Learning Repository - Car Evaluation Dataset  
**URL**: https://archive.ics.uci.edu/dataset/19/car+evaluation

### Veri Seti Özellikleri:
- **Toplam Örnek Sayısı**: 1,728
- **Özellik Sayısı**: 6 (tümü kategorik)
- **Sınıf Sayısı**: 4
- **Eksik Değer**: Yok

### Özellikler:
1. **buying**: Satın alma fiyatı (vhigh, high, med, low)
2. **maint**: Bakım maliyeti (vhigh, high, med, low)
3. **doors**: Kapı sayısı (2, 3, 4, 5more)
4. **persons**: Kişi kapasitesi (2, 4, more)
5. **lug_boot**: Bagaj boyutu (small, med, big)
6. **safety**: Güvenlik seviyesi (low, med, high)

### Hedef Değişken:
**class**: Araba değerlendirmesi (unacc, acc, good, vgood)

## Sınıf Dağılımı

Veri setinde önemli bir sınıf dengesizliği bulunmaktadır:

| Sınıf | Sayı | Yüzde |
|-------|------|-------|
| unacc | 1,210 | %70.0 |
| acc   | 384   | %22.2 |
| good  | 69    | %4.0  |
| vgood | 65    | %3.8  |

## Model Karşılaştırması

Toplam 7 farklı makine öğrenmesi algoritması test edildi:

| Model | Test Doğruluğu | CV Ortalama | CV Std |
|-------|---------------|-------------|---------|
| **Decision Tree** | **98.55%** | **97.10%** | **±1.41%** |
| Random Forest | 98.27% | 96.31% | ±1.29% |
| Gradient Boosting | 97.98% | 97.97% | ±0.93% |
| SVM | 92.77% | 88.71% | ±1.00% |
| K-Nearest Neighbors | 90.75% | 88.28% | ±1.47% |
| Logistic Regression | 69.08% | 69.54% | ±0.65% |
| Naive Bayes | 64.16% | 64.62% | ±2.71% |

## En İyi Model: Decision Tree

**Test Doğruluğu**: 98.55%

### Detaylı Performans Metrikleri:

| Sınıf | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| acc   | 0.95      | 0.99   | 0.97     | 77      |
| good  | 0.93      | 1.00   | 0.97     | 14      |
| unacc | 1.00      | 0.99   | 1.00     | 242     |
| vgood | 1.00      | 0.85   | 0.92     | 13      |

**Genel Doğruluk**: 99%  
**Macro Ortalama**: Precision: 0.97, Recall: 0.96, F1-Score: 0.96  
**Weighted Ortalama**: Precision: 0.99, Recall: 0.99, F1-Score: 0.99

## Hiperparametre Optimizasyonu

Decision Tree modeli için hiperparametre optimizasyonu yapıldı:

**En İyi Parametreler**:
- criterion: 'entropy'
- max_depth: None
- min_samples_leaf: 1
- min_samples_split: 2

**Optimize Edilmiş Model Doğruluğu**: 97.98%

## Önemli Bulgular

1. **Çok Yüksek Performans**: Decision Tree modeli %98.55 test doğruluğu ile mükemmel performans gösterdi.

2. **Kategorik Veri İçin Uygunluk**: Tree-based modeller (Decision Tree, Random Forest, Gradient Boosting) kategorik veriler için en iyi performansı sergiledi.

3. **Sınıf Dengesizliği Etkisi**: %70 oranında 'unacc' sınıfına rağmen, model tüm sınıfları başarıyla ayırt edebildi.

4. **Model Kararlılığı**: Cross-validation skorları model performansının kararlı olduğunu gösteriyor.

## Teknik Detaylar

### Veri Önişleme:
- Label Encoding kullanıldı (kategorik veriler için)
- Train-test split: %80-%20
- Stratified sampling (sınıf dengesini korumak için)

### Model Eğitimi:
- 5-fold Cross Validation
- Grid Search ile hiperparametre optimizasyonu
- Multiple metric evaluation (accuracy, precision, recall, f1-score)

## Üretilen Dosyalar

1. **car_evaluation_model.py**: Ana model kodu
2. **car_evaluation_model_report.txt**: Detaylı analiz raporu
3. **model_comparison_results.csv**: Model karşılaştırma sonuçları
4. **car_evaluation_analysis.png**: Veri analizi görselleştirmeleri
5. **car_evaluation_heatmaps.png**: Sınıf-özellik ilişki haritaları
6. **model_results.png**: Model performans görselleştirmeleri
7. **requirements.txt**: Gerekli Python paketleri

## Sonuç ve Öneriler

1. **Decision Tree modeli** bu veri seti için en uygun çözümdür.
2. Model, otomobil değerlendirmesi için **%98.55 doğrulukla** güvenilir tahminler yapabilir.
3. Modelin yorumlanabilirliği yüksektir ve karar süreçleri açıkça izlenebilir.
4. Üretim ortamında kullanıma hazırdır.

## Gelecek Çalışmalar

1. Daha fazla özellik mühendisliği
2. Ensemble methods ile performans artırımı
3. Yeni veri ile model validasyonu
4. Web uygulaması geliştirme

---

**Analiz Tarihi**: 30 Eylül 2025  
**Kullanılan Teknolojiler**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn  
**Veri Kaynağı**: UCI Machine Learning Repository