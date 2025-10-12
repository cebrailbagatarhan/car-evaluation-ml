# -*- coding: utf-8 -*-
"""
GELIŞMIŞ MODEL GÖRSELLEŞTİRME SİSTEMİ
Karmaşık ve Detaylı Model Çıktı Analizi

Bu sistem, makine öğrenmesi modellerinin en detaylı analizini sunar:
- Confusion Matrix ile detaylı hata analizi
- ROC Curves ve AUC skorları
- Learning Curves ve model kararlılığı
- Feature Interactions (özellik etkileşimleri)
- Prediction Confidence (tahmin güvenilirliği)
- 3D Feature Space Visualization
- Interactive Decision Boundaries
- Model Uncertainty Analysis
"""

# Standart kütüphaneler
import json
import warnings

# Veri işleme kütüphaneleri
import numpy as np
import pandas as pd
import joblib

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Scikit-learn modelleri
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Scikit-learn araçları
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                           precision_recall_curve, accuracy_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, cohen_kappa_score)
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

# Uyarıları kapat
warnings.filterwarnings('ignore')

# 3D görselleştirme için Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
    print("✅ Plotly kütüphanesi yüklendi - 3D görselleştirmeler kullanılabilir")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly kütüphanesi bulunamadı - 3D görselleştirmeler devre dışı")

# Türkçe font desteği - Güvenli font ayarları
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

class AdvancedModelVisualizer:
    def __init__(self, data_path='car.data'):
        """Gelişmiş model görselleştirici başlatılıyor"""
        self.data_path = data_path
        self.models = {}
        self.model_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.class_names = []
        self.encoders = {}
        
        # Renkler ve stiller
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        self.model_colors = {
            'Decision Tree': '#FF6B6B',
            'Random Forest': '#4ECDC4', 
            'Gradient Boosting': '#45B7D1',
            'SVM': '#96CEB4',
            'Logistic Regression': '#FFEAA7',
            'Naive Bayes': '#DDA0DD',
            'K-Nearest Neighbors': '#98D8C8'
        }
        
        print("🚀 Gelişmiş Model Görselleştirici başlatıldı!")
        
    def load_and_prepare_data(self):
        """Veri yükleme ve hazırlama"""
        print("\n📊 Veri yükleniyor ve hazırlanıyor...")
        
        # Sütun isimleri
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        
        # Veri yükleme
        df = pd.read_csv(self.data_path, names=columns)
        
        # Özellik isimleri
        self.feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        
        # Label encoding
        for column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.encoders[column] = le
            
        # Sınıf isimlerini kaydet
        self.class_names = list(self.encoders['class'].classes_)
        
        # Özellikler ve hedef değişken
        X = df[self.feature_names]
        y = df['class']
        
        # Train-test ayırma
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Veri hazırlandı: {X.shape[0]} örnek, {X.shape[1]} özellik")
        print(f"   Eğitim seti: {self.X_train.shape[0]} örnek")
        print(f"   Test seti: {self.X_test.shape[0]} örnek")
        print(f"   Sınıflar: {self.class_names}")
        
    def train_all_models(self):
        """Tüm modelleri eğit"""
        print("\n🎯 Tüm modeller eğitiliyor...")
        
        # Model tanımları
        model_configs = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        self.model_names = list(model_configs.keys())
        
        # Modelleri eğit
        for name, model in model_configs.items():
            print(f"   🔄 {name} eğitiliyor...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
        print("✅ Tüm modeller eğitildi!")
        
    def create_comprehensive_confusion_matrices(self):
        """Detaylı confusion matrix analizi"""
        print("\n🔍 Detaylı Confusion Matrix analizi oluşturuluyor...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Normalleştirilmiş confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Görselleştirme
            ax = axes[idx]
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{name}\nAccuracy: {accuracy_score(self.y_test, y_pred):.3f}', 
                        fontsize=12, fontweight='bold')
            
            # Metin ekleme
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                           ha="center", va="center", fontsize=10,
                           color="white" if cm_normalized[i, j] > thresh else "black")
            
            ax.set_xlabel('Tahmin Edilen')
            ax.set_ylabel('Gerçek')
            ax.set_xticks(range(len(self.class_names)))
            ax.set_yticks(range(len(self.class_names)))
            ax.set_xticklabels(self.class_names, rotation=45)
            ax.set_yticklabels(self.class_names)
            
        # Son subplot'u gizle
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("   💾 comprehensive_confusion_matrices.png kaydedildi")
        
    def create_roc_curves_analysis(self):
        """ROC Curves ve AUC analizi"""
        print("\n📈 ROC Curves ve AUC analizi oluşturuluyor...")
        
        # Çok sınıflı ROC için One-vs-Rest yaklaşımı
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Tüm modeller için ortalama ROC
        colors = iter(self.colors)
        for name, model in self.models.items():
            color = next(colors)
            
            # Probability tahminleri
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
            else:
                # SVM için decision_function kullan
                y_scores = model.decision_function(self.X_test)
                y_proba = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)
            
            # Her sınıf için ROC hesapla
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(len(self.class_names)):
                fpr[i], tpr[i], _ = roc_curve((self.y_test == i).astype(int), y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Micro-average ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(
                pd.get_dummies(self.y_test).values.ravel(),
                y_proba.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Ortalama AUC'yi çiz
            ax1.plot(fpr["micro"], tpr["micro"], color=color, lw=2, 
                    label=f'{name} (AUC = {roc_auc["micro"]:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Tüm Modeller')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. En iyi model için sınıf bazlı ROC
        best_model_name = max(self.models.keys(), 
                             key=lambda x: accuracy_score(self.y_test, self.models[x].predict(self.X_test)))
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(self.X_test)
        else:
            y_scores = best_model.decision_function(self.X_test)
            y_proba = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)
        
        colors = iter(self.colors)
        for i, class_name in enumerate(self.class_names):
            color = next(colors)
            fpr, tpr, _ = roc_curve((self.y_test == i).astype(int), y_proba[:, i])
            roc_auc_score = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'ROC Curves - {best_model_name} (Sınıf Bazlı)')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        colors = iter(self.colors)
        for name, model in self.models.items():
            color = next(colors)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
            else:
                y_scores = model.decision_function(self.X_test)
                y_proba = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)
            
            # Micro-average Precision-Recall
            precision_micro, recall_micro, _ = precision_recall_curve(
                pd.get_dummies(self.y_test).values.ravel(),
                y_proba.ravel()
            )
            avg_precision = auc(recall_micro, precision_micro)
            
            ax3.plot(recall_micro, precision_micro, color=color, lw=2,
                    label=f'{name} (AP = {avg_precision:.3f})')
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend(loc="lower left")
        ax3.grid(True, alpha=0.3)
        
        # 4. AUC Karşılaştırma Bar Chart
        model_aucs = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
            else:
                y_scores = model.decision_function(self.X_test)
                y_proba = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)
            
            fpr_micro, tpr_micro, _ = roc_curve(
                pd.get_dummies(self.y_test).values.ravel(),
                y_proba.ravel()
            )
            auc_score = auc(fpr_micro, tpr_micro)
            model_aucs.append(auc_score)
        
        bars = ax4.bar(range(len(self.models)), model_aucs, 
                      color=self.colors[:len(self.models)])
        ax4.set_xlabel('Modeller')
        ax4.set_ylabel('AUC Score')
        ax4.set_title('Model AUC Karşılaştırması')
        ax4.set_xticks(range(len(self.models)))
        ax4.set_xticklabels(self.model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Bar üzerine değerleri yaz
        for i, (bar, auc_val) in enumerate(zip(bars, model_aucs)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('roc_curves_analysis.png', dpi=300, bbox_inches='tight')
        print("   💾 roc_curves_analysis.png kaydedildi")
        
    def create_learning_curves(self):
        """Learning curves analizi"""
        print("\n📚 Learning Curves analizi oluşturuluyor...")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # Learning curve hesapla
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
            )
            
            # Ortalama ve standart sapma
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Görselleştirme
            ax.plot(train_sizes, train_mean, 'o-', color=self.model_colors[name], 
                   label='Eğitim Skoru', linewidth=2, markersize=6)
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color=self.model_colors[name])
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', 
                   label='Doğrulama Skoru', linewidth=2, markersize=6)
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='red')
            
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Eğitim Seti Boyutu')
            ax.set_ylabel('Doğruluk Skoru')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.05)
            
        # Son subplot'u gizle
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('learning_curves_analysis.png', dpi=300, bbox_inches='tight')
        print("   💾 learning_curves_analysis.png kaydedildi")
        
    def create_feature_importance_analysis(self):
        """Detaylı feature importance analizi"""
        print("\n🎯 Detaylı Feature Importance analizi oluşturuluyor...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Tüm modeller için feature importance
        importance_data = {}
        
        # Tree-based modeller için built-in importance
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
                
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=self.feature_names)
            importance_df.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Feature Importance - Tree-based Modeller')
            ax1.set_xlabel('Özellikler')
            ax1.set_ylabel('Importance Score')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3, axis='y')
            
        # 2. Permutation importance (en iyi model için)
        best_model_name = max(self.models.keys(), 
                             key=lambda x: accuracy_score(self.y_test, self.models[x].predict(self.X_test)))
        best_model = self.models[best_model_name]
        
        perm_importance = permutation_importance(best_model, self.X_test, self.y_test, 
                                               n_repeats=10, random_state=42)
        
        # Permutation importance görselleştirme
        indices = np.argsort(perm_importance.importances_mean)[::-1]
        
        ax2.bar(range(len(self.feature_names)), 
               perm_importance.importances_mean[indices],
               yerr=perm_importance.importances_std[indices],
               color=self.colors[:len(self.feature_names)])
        ax2.set_title(f'Permutation Importance - {best_model_name}')
        ax2.set_xlabel('Özellikler')
        ax2.set_ylabel('Importance Score')
        ax2.set_xticks(range(len(self.feature_names)))
        ax2.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Feature correlation heatmap
        corr_matrix = self.X_train.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax3, cbar_kws={"shrink": .8})
        ax3.set_title('Özellikler Arası Korelasyon')
        
        # 4. Model complexity vs performance
        complexities = []
        performances = []
        model_labels = []
        
        for name, model in self.models.items():
            # Model karmaşıklığını tahmin et
            if hasattr(model, 'n_estimators'):
                complexity = model.n_estimators
            elif hasattr(model, 'max_depth') and model.max_depth:
                complexity = model.max_depth * 10
            elif name == 'SVM':
                complexity = 50
            elif name == 'Logistic Regression':
                complexity = 20
            elif name == 'Naive Bayes':
                complexity = 10
            else:
                complexity = 30
                
            complexities.append(complexity)
            performances.append(accuracy_score(self.y_test, model.predict(self.X_test)))
            model_labels.append(name)
        
        scatter = ax4.scatter(complexities, performances, 
                            c=range(len(self.models)), cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        
        for i, label in enumerate(model_labels):
            ax4.annotate(label, (complexities[i], performances[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Model Karmaşıklığı (Tahmini)')
        ax4.set_ylabel('Test Doğruluğu')
        ax4.set_title('Model Karmaşıklığı vs Performans')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("   💾 feature_importance_analysis.png kaydedildi")
        
    def create_prediction_confidence_analysis(self):
        """Tahmin güvenilirlik analizi"""
        print("\n🎲 Tahmin güvenilirlik analizi oluşturuluyor...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_count = 0
        for name, model in self.models.items():
            if model_count >= 6:  # Maksimum 6 model göster
                break
                
            ax = axes[model_count]
            
            # Probability tahminleri al
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
                confidence = np.max(y_proba, axis=1)
            else:
                # SVM için decision function kullan
                y_scores = model.decision_function(self.X_test)
                if len(y_scores.shape) == 1:  # Binary classification
                    confidence = np.abs(y_scores)
                else:  # Multi-class
                    confidence = np.max(y_scores, axis=1)
                # Normalize et
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            # Tahminleri al
            y_pred = model.predict(self.X_test)
            
            # Doğru ve yanlış tahminleri ayır
            correct_mask = (y_pred == self.y_test)
            correct_conf = confidence[correct_mask]
            incorrect_conf = confidence[~correct_mask]
            
            # Histogram çiz
            ax.hist(correct_conf, bins=20, alpha=0.7, color='green', 
                   label=f'Doğru ({len(correct_conf)})', density=True)
            ax.hist(incorrect_conf, bins=20, alpha=0.7, color='red', 
                   label=f'Yanlış ({len(incorrect_conf)})', density=True)
            
            ax.set_title(f'{name}\nOrtalama Güven: {confidence.mean():.3f}')
            ax.set_xlabel('Tahmin Güveni')
            ax.set_ylabel('Yoğunluk')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            model_count += 1
        
        # Boş subplot'ları gizle
        for i in range(model_count, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
        print("   💾 prediction_confidence_analysis.png kaydedildi")
        
    def create_model_performance_heatmap(self):
        """Model performans ısı haritası"""
        print("\n🌡️ Model performans ısı haritası oluşturuluyor...")
        
        # Performans metrikleri hesapla
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Kappa']
        performance_data = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            # Metrikler hesapla
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(self.y_test, y_pred)
            kappa = cohen_kappa_score(self.y_test, y_pred)
            
            performance_data.append([acc, prec, rec, f1, mcc, kappa])
        
        # DataFrame oluştur
        performance_df = pd.DataFrame(
            performance_data, 
            index=self.model_names, 
            columns=metrics
        )
        
        # Heatmap oluştur
        plt.figure(figsize=(12, 8))
        sns.heatmap(performance_df, annot=True, cmap='RdYlGn', 
                   center=0.5, square=True, cbar_kws={"shrink": .8})
        plt.title('Model Performans Ísı Haritası')
        plt.ylabel('Modeller')
        plt.xlabel('Performans Metrikleri')
        plt.tight_layout()
        plt.savefig('model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        print("   💾 model_performance_heatmap.png kaydedildi")
        
        return performance_df
        
    def create_3d_visualizations(self):
        """3D görselleştirmeler (Plotly ile)"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly yüklü değil - 3D görselleştirmeler atlanıyor")
            return
            
        print("\n🌐 3D görselleştirmeler oluşturuluyor...")
        
        # En iyi 3 özelliği seç (feature importance'a göre)
        best_model_name = max(self.models.keys(), 
                             key=lambda x: accuracy_score(self.y_test, self.models[x].predict(self.X_test)))
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importance_indices = np.argsort(best_model.feature_importances_)[-3:]
            selected_features = [self.feature_names[i] for i in importance_indices]
        else:
            # Rastgele 3 özellik seç
            selected_features = self.feature_names[:3]
        
        # 3D scatter plot oluştur
        X_3d = self.X_test[selected_features]
        
        fig = go.Figure(data=go.Scatter3d(
            x=X_3d.iloc[:, 0],
            y=X_3d.iloc[:, 1], 
            z=X_3d.iloc[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=self.y_test,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sınıf")
            ),
            text=[f'Sınıf: {self.class_names[int(y)]}' for y in self.y_test],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{selected_features[0]}: %{{x}}<br>' +
                         f'{selected_features[1]}: %{{y}}<br>' +
                         f'{selected_features[2]}: %{{z}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Feature Space Visualization<br>En Önemli 3 Özellik: {", ".join(selected_features)}',
            scene=dict(
                xaxis_title=selected_features[0],
                yaxis_title=selected_features[1],
                zaxis_title=selected_features[2]
            ),
            width=800,
            height=600
        )
        
        # HTML olarak kaydet
        fig.write_html("3d_feature_space.html")
        print("   💾 3d_feature_space.html kaydedildi")
        
        # Model karşılaştırma 3D bar chart
        performance_data = []
        model_names = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            
            performance_data.append([acc, prec, rec])
            model_names.append(name)
        
        performance_array = np.array(performance_data)
        
        fig2 = go.Figure(data=[
            go.Bar(name='Accuracy', x=model_names, y=performance_array[:, 0], offsetgroup=1),
            go.Bar(name='Precision', x=model_names, y=performance_array[:, 1], offsetgroup=2),
            go.Bar(name='Recall', x=model_names, y=performance_array[:, 2], offsetgroup=3)
        ])
        
        fig2.update_layout(
            title='3D Model Performans Karşılaştırması',
            xaxis_title='Modeller',
            yaxis_title='Skor',
            barmode='group',
            width=1000,
            height=600
        )
        
        fig2.write_html("3d_model_comparison.html")
        print("   💾 3d_model_comparison.html kaydedildi")
        
    def create_comprehensive_report(self):
        """Kapsamlı analiz raporu oluştur"""
        print("\n📋 Kapsamlı analiz raporu oluşturuluyor...")
        
        report = []
        report.append("=" * 80)
        report.append("KARMA\u015eIK VE DETAYLI MODEL ANALIZ RAPORU")
        report.append("=" * 80)
        report.append("")
        
        # Veri seti bilgileri
        report.append("📊 VERİ SETİ BİLGİLERİ:")
        report.append(f"   • Toplam örnek sayısı: {len(self.X_train) + len(self.X_test)}")
        report.append(f"   • Eğitim seti: {len(self.X_train)} örnek")
        report.append(f"   • Test seti: {len(self.X_test)} örnek")
        report.append(f"   • Özellik sayısı: {len(self.feature_names)}")
        report.append(f"   • Sınıf sayısı: {len(self.class_names)}")
        report.append(f"   • Sınıflar: {', '.join(self.class_names)}")
        report.append("")
        
        # Model performansları
        report.append("🎯 MODEL PERFORMANSLARI:")
        report.append("-" * 50)
        
        performance_metrics = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            report.append(f"{name:20} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            performance_metrics.append((name, acc, prec, rec, f1))
        
        # En iyi model
        best_model = max(performance_metrics, key=lambda x: x[1])
        report.append("")
        report.append(f"🏆 EN İYİ MODEL: {best_model[0]} (Accuracy: {best_model[1]:.4f})")
        report.append("")
        
        # Feature importance (eğer varsa)
        best_model_obj = self.models[best_model[0]]
        if hasattr(best_model_obj, 'feature_importances_'):
            report.append("🎯 ÖZELLİK ÖNEMLİLİK SIRALAMASI:")
            report.append("-" * 40)
            
            importance_pairs = list(zip(self.feature_names, best_model_obj.feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs, 1):
                report.append(f"{i:2d}. {feature:15} : {importance:.4f}")
            report.append("")
        
        # Sınıf bazlı analiz
        report.append("📈 SINIF BAZLI ANALIZ:")
        report.append("-" * 30)
        
        y_pred_best = best_model_obj.predict(self.X_test)
        class_report = classification_report(self.y_test, y_pred_best, 
                                           target_names=self.class_names, 
                                           output_dict=True)
        
        for class_name in self.class_names:
            metrics = class_report[class_name]
            report.append(f"{class_name:10} | Prec: {metrics['precision']:.3f} | "
                         f"Rec: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | "
                         f"Support: {int(metrics['support'])}")
        
        report.append("")
        
        # Model karmaşıklığı analizi
        report.append("🔧 MODEL KARMAŞIKLIK ANALİZİ:")
        report.append("-" * 35)
        
        for name, model in self.models.items():
            complexity_info = self._analyze_model_complexity(model)
            report.append(f"{name:20} | {complexity_info}")
        
        report.append("")
        
        # Öneriler
        report.append("💡 ÖNERİLER VE DEĞERLENDİRME:")
        report.append("-" * 40)
        
        if best_model[1] > 0.95:
            report.append("✅ Model performansı mükemmel seviyede")
        elif best_model[1] > 0.90:
            report.append("✅ Model performansı çok iyi seviyede")
        elif best_model[1] > 0.80:
            report.append("⚠️ Model performansı orta seviyede - iyileştirme gerekebilir")
        else:
            report.append("❌ Model performansı düşük - ciddi iyileştirme gerekli")
        
        report.append("")
        report.append("🔍 OLUŞTURULAN GÖRSELLEŞTİRMELER:")
        report.append("   • comprehensive_confusion_matrices.png")
        report.append("   • roc_curves_analysis.png") 
        report.append("   • learning_curves_analysis.png")
        report.append("   • feature_importance_analysis.png")
        report.append("   • prediction_confidence_analysis.png")
        report.append("   • model_performance_heatmap.png")
        
        if PLOTLY_AVAILABLE:
            report.append("   • 3d_feature_space.html")
            report.append("   • 3d_model_comparison.html")
        
        report.append("")
        report.append("=" * 80)
        
        # Raporu kaydet
        with open('comprehensive_model_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("   💾 comprehensive_model_analysis_report.txt kaydedildi")
        
        return '\n'.join(report)
    
    def _analyze_model_complexity(self, model):
        """Model karmaşıklığını analiz et"""
        if hasattr(model, 'n_estimators'):
            return f"Ensemble - {model.n_estimators} estimator"
        elif hasattr(model, 'max_depth'):
            depth = model.max_depth if model.max_depth else "Unlimited"
            return f"Tree - Max depth: {depth}"
        elif 'SVM' in str(type(model)):
            return f"SVM - Kernel: {getattr(model, 'kernel', 'unknown')}"
        elif 'Logistic' in str(type(model)):
            return f"Linear - {getattr(model, 'solver', 'unknown')} solver"
        elif 'Naive' in str(type(model)):
            return "Probabilistic - Gaussian assumption"
        elif 'Neighbors' in str(type(model)):
            return f"Distance-based - {getattr(model, 'n_neighbors', 5)} neighbors"
        else:
            return "Unknown complexity"
    
    def run_complete_analysis(self):
        """Tüm analizi çalıştır"""
        print("🚀 KARMA\u015eIK VE DETAYLI MODEL ANALİZİ BA\u015eLIYOR...")
        print("=" * 60)
        
        # Veri hazırlama ve model eğitimi
        self.load_and_prepare_data()
        self.train_all_models()
        
        # Tüm görselleştirmeleri oluştur
        self.create_comprehensive_confusion_matrices()
        self.create_roc_curves_analysis()
        self.create_learning_curves()
        self.create_feature_importance_analysis()
        self.create_prediction_confidence_analysis()
        performance_df = self.create_model_performance_heatmap()
        
        # 3D görselleştirmeler (eğer Plotly varsa)
        self.create_3d_visualizations()
        
        # Kapsamlı rapor
        report = self.create_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("🎉 TÜM KARMA\u015eIK ANALIZLER TAMAMLANDI!")
        print("=" * 60)
        
        print("\n📊 OLUŞTURULAN DOSYALAR:")
        print("   🖼️ comprehensive_confusion_matrices.png")
        print("   📈 roc_curves_analysis.png")
        print("   📚 learning_curves_analysis.png") 
        print("   🎯 feature_importance_analysis.png")
        print("   🎲 prediction_confidence_analysis.png")
        print("   🌡️ model_performance_heatmap.png")
        print("   📋 comprehensive_model_analysis_report.txt")
        
        if PLOTLY_AVAILABLE:
            print("   🌐 3d_feature_space.html")
            print("   📊 3d_model_comparison.html")
        
        print(f"\n🏆 EN İYİ MODEL: {performance_df.loc[:, 'Accuracy'].idxmax()}")
        print(f"   Doğruluk: {performance_df.loc[:, 'Accuracy'].max():.4f}")
        
        return performance_df, report

if __name__ == "__main__":
    # Ana analizi çalıştır
    analyzer = AdvancedModelVisualizer()
    performance_df, report = analyzer.run_complete_analysis()
    
    print("\n" + "🎯 KARMA\u015eIK ANALIZ SONUÇLARI" + "\n")
    print(performance_df.round(4))