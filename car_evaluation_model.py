"""
Car Evaluation Dataset - Machine Learning Model
===============================================

Bu script, UCI Car Evaluation veri setini kullanarak makine öğrenmesi modelleri oluşturur.
Veri seti, otomobillerin kabul edilebilirliğini çeşitli faktörlere göre değerlendirir.

Veri Seti Özellikleri:
- 1728 örnek
- 6 özellik (tümü kategorik)
- 4 sınıf (unacc, acc, good, vgood)

Özellikler:
1. buying: satın alma fiyatı (vhigh, high, med, low)
2. maint: bakım maliyeti (vhigh, high, med, low)
3. doors: kapı sayısı (2, 3, 4, 5more)
4. persons: kişi kapasitesi (2, 4, more)
5. lug_boot: bagaj boyutu (small, med, big)
6. safety: güvenlik (low, med, high)

Hedef: class (unacc, acc, good, vgood)
"""

# Standart kütüphaneler
import warnings

# Veri işleme kütüphaneleri
import pandas as pd
import numpy as np

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Scikit-learn model seçimi ve değerlendirme
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Scikit-learn modelleri
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Matplotlib Türkçe karakter desteği için - güvenli font ayarları
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Arial Unicode MS' in available_fonts:
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    elif 'DejaVu Sans' in available_fonts:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    else:
        plt.rcParams['font.family'] = 'sans-serif'
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Fallback

plt.rcParams['axes.unicode_minus'] = False

class CarEvaluationModel:
    def __init__(self, data_path):
        """
        Car Evaluation Model sınıfı başlatıcısı
        
        Args:
            data_path (str): Veri dosyasının yolu
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.models = {}
        self.model_scores = {}
        
        # Özellik ve sınıf adları
        self.feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        self.class_name = 'class'
        
    def load_data(self):
        """Veri setini yükle ve temizle"""
        print("Veri seti yükleniyor...")
        
        # Veri setini yükle
        self.df = pd.read_csv(self.data_path, names=self.feature_names + [self.class_name])
        
        print(f"Veri seti başarıyla yüklendi: {self.df.shape}")
        print("\nVeri setinin ilk 5 satırı:")
        print(self.df.head())
        
        return self.df
    
    def explore_data(self):
        """Veri setini keşfet ve analiz et"""
        print("\n" + "="*50)
        print("VERİ SETİ ANALİZİ")
        print("="*50)
        
        # Temel bilgiler
        print(f"Veri seti boyutu: {self.df.shape}")
        print(f"Özellik sayısı: {len(self.feature_names)}")
        print(f"Eksik değer sayısı: {self.df.isnull().sum().sum()}")
        
        # Sınıf dağılımı
        print("\nSınıf Dağılımı:")
        class_dist = self.df[self.class_name].value_counts()
        print(class_dist)
        print("\nSınıf Yüzdeleri:")
        print(self.df[self.class_name].value_counts(normalize=True) * 100)
        
        # Her özellik için benzersiz değerler
        print("\nÖzellik Değerleri:")
        for feature in self.feature_names:
            unique_values = self.df[feature].unique()
            print(f"{feature}: {unique_values}")
            
        return class_dist
    
    def visualize_data(self):
        """Veri setini görselleştir"""
        print("\nVeri görselleştirme yapılıyor...")
        
        # Figure boyutunu ayarla
        plt.style.use('default')
        
        # 1. Sınıf dağılımı
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Car Evaluation Dataset - Veri Analizi', fontsize=16, fontweight='bold')
        
        # Sınıf dağılımı pasta grafiği
        class_counts = self.df[self.class_name].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Sınıf Dağılımı')
        
        # Sınıf dağılımı bar grafiği
        class_counts.plot(kind='bar', ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title('Sınıf Sayıları')
        axes[0, 1].set_xlabel('Sınıflar')
        axes[0, 1].set_ylabel('Sayı')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Özellik dağılımları - sadece 4 özellik için (2x2 grid'in kalan yerlerini kullan)
        features_to_plot = ['buying', 'maint', 'doors', 'persons']
        plot_positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, feature in enumerate(features_to_plot):
            row, col = plot_positions[i]
            feature_counts = self.df[feature].value_counts()
            feature_counts.plot(kind='bar', ax=axes[row, col], color='lightcoral')
            axes[row, col].set_title(f'{feature.title()} Dağılımı')
            axes[row, col].set_xlabel(feature.title())
            axes[row, col].set_ylabel('Sayı')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('car_evaluation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Sınıflara göre özellik dağılımı heatmap
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sınıflara Göre Özellik Dağılımları', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.feature_names):
            row = i // 3
            col = i % 3
            
            # Crosstab oluştur
            ct = pd.crosstab(self.df[feature], self.df[self.class_name], normalize='index') * 100
            
            # Heatmap
            sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues', ax=axes[row, col])
            axes[row, col].set_title(f'{feature.title()} vs Class (%)')
            axes[row, col].set_xlabel('Sınıf')
            axes[row, col].set_ylabel(feature.title())
        
        plt.tight_layout()
        plt.savefig('car_evaluation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Veriyi önişleme"""
        print("\nVeri önişleme yapılıyor...")
        
        # Label encoding için her özellik ve sınıf için encoder oluştur
        df_encoded = self.df.copy()
        
        # Özellikleri encode et
        for feature in self.feature_names:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(self.df[feature])
            self.label_encoders[feature] = le
            print(f"{feature}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Sınıfları encode et
        le_class = LabelEncoder()
        df_encoded[self.class_name] = le_class.fit_transform(self.df[self.class_name])
        self.label_encoders[self.class_name] = le_class
        print(f"class: {dict(zip(le_class.classes_, le_class.transform(le_class.classes_)))}")
        
        # X ve y'yi ayır
        self.X = df_encoded[self.feature_names].values
        self.y = df_encoded[self.class_name].values
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Eğitim seti boyutu: {self.X_train.shape}")
        print(f"Test seti boyutu: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Çoklu model eğitimi"""
        print("\n" + "="*50)
        print("MODEL EĞİTİMİ")
        print("="*50)
        
        # Model tanımları
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Her modeli eğit ve değerlendir
        for name, model in models.items():
            print(f"\n{name} eğitiliyor...")
            
            # Modeli eğit
            model.fit(self.X_train, self.y_train)
            
            # Tahmin yap
            y_pred = model.predict(self.X_test)
            
            # Doğruluk hesapla
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation skoru
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Sonuçları kaydet
            self.models[name] = model
            self.model_scores[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"Test Doğruluğu: {accuracy:.4f}")
            print(f"CV Doğruluğu: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.models, self.model_scores
    
    def evaluate_models(self):
        """Model değerlendirmesi"""
        print("\n" + "="*50)
        print("MODEL DEĞERLENDİRMESİ")
        print("="*50)
        
        # Sonuçları DataFrame'e dönüştür
        results_df = pd.DataFrame({
            'Model': list(self.model_scores.keys()),
            'Test Accuracy': [scores['accuracy'] for scores in self.model_scores.values()],
            'CV Mean': [scores['cv_mean'] for scores in self.model_scores.values()],
            'CV Std': [scores['cv_std'] for scores in self.model_scores.values()]
        })
        
        # Sırala
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        print("\nModel Performans Karşılaştırması:")
        print(results_df.to_string(index=False))
        
        # En iyi modeli bul
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        print(f"\nEn İyi Model: {best_model_name}")
        print(f"En İyi Test Doğruluğu: {results_df.iloc[0]['Test Accuracy']:.4f}")
        
        # En iyi model için detaylı rapor
        y_pred_best = self.model_scores[best_model_name]['predictions']
        
        print(f"\n{best_model_name} - Detaylı Sınıflandırma Raporu:")
        class_names = self.label_encoders[self.class_name].classes_
        print(classification_report(self.y_test, y_pred_best, 
                                  target_names=class_names))
        
        return results_df, best_model_name, best_model
    
    def visualize_decision_tree(self, model_name="Decision Tree"):
        """Karar ağacının yapısını görselleştir"""
        if model_name not in self.models:
            print(f"❌ {model_name} modeli bulunamadı.")
            return
        
        model = self.models[model_name]
        
        # Karar ağacını görselleştir
        plt.figure(figsize=(20, 15))
        plot_tree(
            model,
            feature_names=self.feature_names,
            class_names=self.label_encoders[self.class_name].classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=4  # Okunabilirlik için sınırlı derinlik
        )
        plt.title(f'{model_name} - Karar Ağacı Yapısı', fontsize=16, fontweight='bold')
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Ağaç kurallarını text olarak göster
        print(f"\n📝 {model_name.upper()} KURALLARI:")
        print("=" * 40)
        tree_rules = export_text(
            model,
            feature_names=self.feature_names,
            show_weights=True,
            max_depth=5
        )
        print(tree_rules)
        
        # Dosyaya kaydet
        with open(f'{model_name.lower().replace(" ", "_")}_rules.txt', 'w', encoding='utf-8') as f:
            f.write(f"{model_name.upper()} KARAR KURALLARI\n")
            f.write("=" * 40 + "\n\n")
            f.write("Özellik Değer Eşlemeleri:\n")
            for feature in self.feature_names:
                values = self.label_encoders[feature].classes_
                mapping = {i: val for i, val in enumerate(values)}
                f.write(f"{feature}: {mapping}\n")
            f.write(f"\nSınıflar: {dict(enumerate(self.label_encoders[self.class_name].classes_))}\n\n")
            f.write("Karar Ağacı Kuralları:\n")
            f.write("-" * 25 + "\n")
            f.write(tree_rules)
        
        print(f"💾 Karar kuralları '{model_name.lower().replace(' ', '_')}_rules.txt' dosyasına kaydedildi")
    
    def visualize_results(self, results_df, best_model_name):
        """Sonuçları görselleştir"""
        print("\nSonuçlar görselleştiriliyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performans Analizi', fontsize=16, fontweight='bold')
        
        # 1. Model performans karşılaştırması
        x_pos = np.arange(len(results_df))
        axes[0, 0].bar(x_pos, results_df['Test Accuracy'], alpha=0.8, color='skyblue')
        axes[0, 0].set_xlabel('Modeller')
        axes[0, 0].set_ylabel('Test Doğruluğu')
        axes[0, 0].set_title('Model Performans Karşılaştırması')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        
        # En iyi modeli vurgula
        best_idx = results_df[results_df['Model'] == best_model_name].index[0]
        axes[0, 0].bar(best_idx, results_df.iloc[best_idx]['Test Accuracy'], 
                      color='orange', alpha=0.8, label='En İyi Model')
        axes[0, 0].legend()
        
        # 2. Confusion Matrix (en iyi model için)
        y_pred_best = self.model_scores[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred_best)
        class_names = self.label_encoders[self.class_name].classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Tahmin Edilen')
        axes[0, 1].set_ylabel('Gerçek')
        
        # 3. CV Scores with error bars
        cv_means = [scores['cv_mean'] for scores in self.model_scores.values()]
        cv_stds = [scores['cv_std'] for scores in self.model_scores.values()]
        
        axes[1, 0].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o-', capsize=5)
        axes[1, 0].set_xlabel('Modeller')
        axes[1, 0].set_ylabel('CV Doğruluğu')
        axes[1, 0].set_title('Cross-Validation Skorları')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (eğer en iyi model destekliyorsa)
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            importances = self.models[best_model_name].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[1, 1].bar(range(len(importances)), importances[indices], color='lightgreen')
            axes[1, 1].set_xlabel('Özellikler')
            axes[1, 1].set_ylabel('Önem Skoru')
            axes[1, 1].set_title(f'Özellik Önemleri - {best_model_name}')
            axes[1, 1].set_xticks(range(len(importances)))
            axes[1, 1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, f'{best_model_name}\nözellik önemlerini\ndesteklemiyor', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Özellik Önemleri')
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self, best_model_name):
        """En iyi model için hiperparametre optimizasyonu"""
        print(f"\n{best_model_name} için hiperparametre optimizasyonu yapılıyor...")
        
        # Model tipine göre parametre grid'i tanımla
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Decision Tree': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if best_model_name in param_grids:
            # Model sınıfını al
            model_classes = {
                'Random Forest': RandomForestClassifier,
                'Decision Tree': DecisionTreeClassifier,
                'SVM': SVC,
                'Gradient Boosting': GradientBoostingClassifier
            }
            
            model = model_classes[best_model_name](random_state=42)
            param_grid = param_grids[best_model_name]
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"En iyi parametreler: {grid_search.best_params_}")
            print(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
            
            # Optimized model ile test et
            optimized_predictions = grid_search.predict(self.X_test)
            optimized_accuracy = accuracy_score(self.y_test, optimized_predictions)
            
            print(f"Optimize edilmiş model test doğruluğu: {optimized_accuracy:.4f}")
            
            return grid_search.best_estimator_, optimized_accuracy
        else:
            print(f"{best_model_name} için hiperparametre optimizasyonu desteklenmiyor.")
            return None, None
    
    def save_results(self, results_df, best_model_name):
        """Sonuçları dosyaya kaydet"""
        print("\nSonuçlar kaydediliyor...")
        
        # Sonuçları CSV olarak kaydet
        results_df.to_csv('model_comparison_results.csv', index=False)
        
        # Detaylı rapor oluştur
        with open('car_evaluation_model_report.txt', 'w', encoding='utf-8') as f:
            f.write("CAR EVALUATION DATASET - MODEL ANALİZİ RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("VERİ SETİ BİLGİLERİ:\n")
            f.write(f"- Toplam örnek sayısı: {len(self.df)}\n")
            f.write(f"- Özellik sayısı: {len(self.feature_names)}\n")
            f.write(f"- Sınıf sayısı: {len(self.df[self.class_name].unique())}\n")
            f.write(f"- Eksik değer: {self.df.isnull().sum().sum()}\n\n")
            
            f.write("SINIF DAĞILIMI:\n")
            class_dist = self.df[self.class_name].value_counts()
            for class_name, count in class_dist.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"- {class_name}: {count} (%{percentage:.1f})\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANSLARI:\n")
            for _, row in results_df.iterrows():
                f.write(f"- {row['Model']}: {row['Test Accuracy']:.4f} "
                       f"(CV: {row['CV Mean']:.4f} ± {row['CV Std']:.4f})\n")
            f.write("\n")
            
            f.write(f"EN İYİ MODEL: {best_model_name}\n")
            f.write(f"TEST DOĞRULUĞU: {results_df.iloc[0]['Test Accuracy']:.4f}\n\n")
            
            # En iyi model için classification report
            y_pred_best = self.model_scores[best_model_name]['predictions']
            class_names = self.label_encoders[self.class_name].classes_
            report = classification_report(self.y_test, y_pred_best, 
                                         target_names=class_names)
            f.write("DETAYLI SINIFLANDIRMA RAPORU:\n")
            f.write(report)
        
        print("Sonuçlar başarıyla kaydedildi:")
        print("- model_comparison_results.csv")
        print("- car_evaluation_model_report.txt")
        print("- car_evaluation_analysis.png")
        print("- car_evaluation_heatmaps.png")
        print("- model_results.png")
    
    def run_complete_analysis(self):
        """Tam analizi çalıştır"""
        print("CAR EVALUATION DATASET - MAKINE ÖĞRENMESİ ANALİZİ")
        print("=" * 60)
        
        # 1. Veri yükleme
        self.load_data()
        
        # 2. Veri keşfi
        self.explore_data()
        
        # 3. Veri görselleştirme
        self.visualize_data()
        
        # 4. Veri önişleme
        self.preprocess_data()
        
        # 5. Model eğitimi
        self.train_models()
        
        # 6. Model değerlendirme
        results_df, best_model_name, best_model = self.evaluate_models()
        
        # 7. Sonuç görselleştirme
        self.visualize_results(results_df, best_model_name)
        
        # 8. Hiperparametre optimizasyonu
        optimized_model, optimized_accuracy = self.hyperparameter_tuning(best_model_name)
        
        # 8.5. Karar ağacı görselleştirmesi (eğer en iyi model Decision Tree ise)
        if "Decision Tree" in best_model_name or "Tree" in best_model_name:
            print("\n🌳 Karar ağacı yapısı görselleştiriliyor...")
            self.visualize_decision_tree(best_model_name)
        
        # 9. Sonuçları kaydetme
        self.save_results(results_df, best_model_name)
        
        print("\n" + "="*60)
        print("ANALİZ TAMAMLANDI!")
        print("="*60)
        
        return results_df, best_model_name, best_model, optimized_model


def main():
    """Ana fonksiyon"""
    # Veri dosyasının yolu
    data_path = "car.data"
    
    # Model oluştur ve analizi çalıştır
    car_model = CarEvaluationModel(data_path)
    results_df, best_model_name, best_model, optimized_model = car_model.run_complete_analysis()
    
    # Özet bilgi
    print(f"\nÖZET:")
    print(f"En iyi model: {best_model_name}")
    print(f"Test doğruluğu: {results_df.iloc[0]['Test Accuracy']:.4f}")
    
    if optimized_model is not None:
        optimized_predictions = optimized_model.predict(car_model.X_test)
        optimized_accuracy = accuracy_score(car_model.y_test, optimized_predictions)
        print(f"Optimize edilmiş model doğruluğu: {optimized_accuracy:.4f}")


if __name__ == "__main__":
    main()