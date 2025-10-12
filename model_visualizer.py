"""
Model Görselleştirme Sistemi
============================

Bu script, eğitilmiş modellerin yapılarını ve şekillerini görselleştirir.
Özellikle karar ağaçlarının detaylı yapısını gösterir.
"""

# Standart kütüphaneler
import os
import pickle
import warnings

# Veri işleme kütüphaneleri
import pandas as pd
import numpy as np

# Görselleştirme kütüphaneleri
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Scikit-learn kütüphaneleri
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Uyarıları kapat
warnings.filterwarnings('ignore')

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

# Opsiyonel kütüphaneler
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️ Graphviz kurulu değil. 'pip install graphviz' ile kurabilirsiniz.")

try:
    from sklearn.inspection import plot_partial_dependence
    PARTIAL_DEP_AVAILABLE = True
except ImportError:
    PARTIAL_DEP_AVAILABLE = False
    print("ℹ️ Partial dependence plots kullanılamıyor (sklearn sürümü eski olabilir)")

class ModelVisualizer:
    def __init__(self):
        """Model görselleştirici sınıfı"""
        self.models = {}
        self.label_encoders = {}
        self.feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        self.class_names = None
        
        # Türkçe özellik adları
        self.feature_names_tr = {
            'buying': 'Satın Alma Fiyatı',
            'maint': 'Bakım Maliyeti',
            'doors': 'Kapı Sayısı',
            'persons': 'Kişi Kapasitesi',
            'lug_boot': 'Bagaj Boyutu',
            'safety': 'Güvenlik'
        }
    
    def load_models_and_data(self):
        """Modelleri ve veriyi yükle"""
        print("📊 Modeller ve veri yükleniyor...")
        
        # Veri setini yükle
        df = pd.read_csv('car.data', names=self.feature_names + ['class'])
        
        # Label encoding
        df_encoded = df.copy()
        for feature in self.feature_names + ['class']:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df[feature])
            self.label_encoders[feature] = le
        
        # Sınıf adlarını al
        self.class_names = self.label_encoders['class'].classes_
        
        # Özellikler ve hedef
        X = df_encoded[self.feature_names].values
        y = df_encoded['class'].values
        
        # Modelleri eğit
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=None,  # Maksimum derinlik - tam detay
                min_samples_split=2,  # Minimum split - en detaylı
                min_samples_leaf=1,  # Minimum leaf - en detaylı
                criterion='entropy'  # Information gain için
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=10,  # Daha fazla ağaç
                max_depth=None,  # Derin ağaçlar
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        }
        
        # Modelleri eğit
        for name, model in self.models.items():
            print(f"🔄 {name} eğitiliyor...")
            model.fit(X, y)
        
        self.X = X
        self.y = y
        self.df = df
        self.df_encoded = df_encoded
        
        print("✅ Modeller ve veri başarıyla yüklendi!")
    
    def visualize_decision_tree_structure(self, save_files=True):
        """Karar ağacının yapısını görselleştir"""
        print("\n🌳 KARAR AĞACI YAPISINI GÖRSELLEŞTİRME")
        print("=" * 50)
        
        dt_model = self.models['Decision Tree']
        
        # 1. TAM DETAYLI Matplotlib görselleştirme
        print("   📊 Tam detaylı ağaç görselleştirmesi oluşturuluyor...")
        plt.figure(figsize=(40, 30))  # Çok daha büyük boyut
        plot_tree(
            dt_model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=None,  # TÜM DEĞERLERİ GÖSTER
            impurity=True,
            proportion=True
        )
        plt.title(f'Karar Ağacı Yapısı - TAM DETAY (Derinlik: {dt_model.tree_.max_depth}, Düğüm: {dt_model.tree_.node_count})', 
                  fontsize=20, fontweight='bold', pad=20)
        
        if save_files:
            plt.savefig('decision_tree_FULL_DETAIL.png', dpi=300, bbox_inches='tight')
            print("💾 TAM DETAYLI ağaç: 'decision_tree_FULL_DETAIL.png'")
        
        plt.show()
        
        # 2. ORTA DETAY görselleştirme (okunabilirlik için)
        print("   📊 Orta detay ağaç görselleştirmesi oluşturuluyor...")
        plt.figure(figsize=(30, 20))
        plot_tree(
            dt_model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=6,  # İlk 6 seviye
            impurity=True
        )
        plt.title('Karar Ağacı Yapısı - İlk 6 Seviye (Okunabilir)', fontsize=16, fontweight='bold', pad=20)
        
        if save_files:
            plt.savefig('decision_tree_MEDIUM_DETAIL.png', dpi=300, bbox_inches='tight')
            print("💾 Orta detay ağaç: 'decision_tree_MEDIUM_DETAIL.png'")
        
        plt.show()
        
        # 3. ÖZET görselleştirme
        print("   📊 Özet ağaç görselleştirmesi oluşturuluyor...")
        plt.figure(figsize=(20, 15))
        plot_tree(
            dt_model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=12,
            max_depth=3  # İlk 3 seviye
        )
        plt.title('Karar Ağacı Yapısı - ÖZET (İlk 3 Seviye)', fontsize=16, fontweight='bold', pad=20)
        
        if save_files:
            plt.savefig('decision_tree_SUMMARY.png', dpi=300, bbox_inches='tight')
            print("💾 Özet ağaç: 'decision_tree_SUMMARY.png'")
        
        plt.show()
        
        # 4. TAM DETAYLI ağaç yapısı (text formatında)
        print("\n📝 KARAR AĞACI METİN YAPISI - TAM DETAY:")
        print("-" * 60)
        tree_rules_full = export_text(
            dt_model,
            feature_names=self.feature_names,
            show_weights=True,
            max_depth=100  # Çok yüksek değer = tüm detay
        )
        
        # İstatistikler
        print(f"\n📊 AĞAÇ İSTATİSTİKLERİ:")
        print(f"   • Toplam Derinlik: {dt_model.tree_.max_depth}")
        print(f"   • Toplam Düğüm Sayısı: {dt_model.tree_.node_count}")
        print(f"   • Yaprak Düğüm Sayısı: {dt_model.tree_.n_leaves}")
        print(f"   • İç Düğüm Sayısı: {dt_model.tree_.node_count - dt_model.tree_.n_leaves}")
        
        # İlk 50 satırı göster (çok uzun olabilir)
        lines = tree_rules_full.split('\n')
        print(f"\n   Toplam {len(lines)} satır kural bulundu.")
        print(f"   İlk 100 satır gösteriliyor...\n")
        print('\n'.join(lines[:100]))
        if len(lines) > 100:
            print(f"\n   ... ve {len(lines) - 100} satır daha (tam detaylar dosyada)")
        
        if save_files:
            # TAM DETAY dosyası
            with open('decision_tree_rules_FULL_DETAIL.txt', 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("KARAR AĞACI KURALLARI - TAM DETAY\n")
                f.write("="*80 + "\n\n")
                
                f.write("📊 AĞAÇ İSTATİSTİKLERİ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Toplam Derinlik: {dt_model.tree_.max_depth}\n")
                f.write(f"Toplam Düğüm Sayısı: {dt_model.tree_.node_count}\n")
                f.write(f"Yaprak Düğüm Sayısı: {dt_model.tree_.n_leaves}\n")
                f.write(f"İç Düğüm Sayısı: {dt_model.tree_.node_count - dt_model.tree_.n_leaves}\n")
                f.write(f"Özellik Sayısı: {len(self.feature_names)}\n")
                f.write(f"Sınıf Sayısı: {len(self.class_names)}\n\n")
                
                f.write("📋 ÖZELLİK DEĞERLERİ:\n")
                f.write("-" * 40 + "\n")
                for feature in self.feature_names:
                    values = self.label_encoders[feature].classes_
                    f.write(f"{feature}: {list(values)}\n")
                f.write(f"\nSınıflar: {list(self.class_names)}\n\n")
                
                f.write("🌳 AĞAÇ YAPISI - TAM DETAY:\n")
                f.write("="*80 + "\n\n")
                f.write(tree_rules_full)
                
            print("💾 TAM DETAY kurallar: 'decision_tree_rules_FULL_DETAIL.txt'")
            
            # ÖZET dosyası (daha okunabilir)
            tree_rules_summary = export_text(
                dt_model,
                feature_names=self.feature_names,
                show_weights=True,
                max_depth=5
            )
            
            with open('decision_tree_rules_SUMMARY.txt', 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("KARAR AĞACI KURALLARI - ÖZET (İlk 5 Seviye)\n")
                f.write("="*80 + "\n\n")
                
                f.write("📊 AĞAÇ İSTATİSTİKLERİ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Gösterilen Derinlik: 5 / {dt_model.tree_.max_depth}\n")
                f.write(f"Toplam Düğüm: {dt_model.tree_.node_count}\n")
                f.write(f"Yaprak Düğüm: {dt_model.tree_.n_leaves}\n\n")
                
                f.write("🌳 AĞAÇ YAPISI - ÖZET:\n")
                f.write("="*80 + "\n\n")
                f.write(tree_rules_summary)
                
            print("💾 ÖZET kurallar: 'decision_tree_rules_SUMMARY.txt'")
        
        # 5. Graphviz ile profesyonel görselleştirme (eğer mevcut ise)
        if GRAPHVIZ_AVAILABLE:
            try:
                print("\n   🎨 Graphviz görselleştirmeleri oluşturuluyor...")
                
                # TAM DETAY Graphviz
                dot_data_full = export_graphviz(
                    dt_model,
                    out_file=None,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    max_depth=None,  # TAM DETAY
                    proportion=True,
                    impurity=True,
                    label='all'
                )
                
                graph_full = graphviz.Source(dot_data_full)
                if save_files:
                    graph_full.render('decision_tree_graphviz_FULL', format='png', cleanup=True)
                    print("💾 Graphviz TAM DETAY PNG: 'decision_tree_graphviz_FULL.png'")
                    graph_full.render('decision_tree_graphviz_FULL', format='svg', cleanup=True)
                    print("💾 Graphviz TAM DETAY SVG: 'decision_tree_graphviz_FULL.svg' (ölçeklenebilir)")
                    graph_full.render('decision_tree_graphviz_FULL', format='pdf', cleanup=True)
                    print("💾 Graphviz TAM DETAY PDF: 'decision_tree_graphviz_FULL.pdf' (yüksek kalite)")
                
                # ORTA DETAY Graphviz (daha okunabilir)
                dot_data_medium = export_graphviz(
                    dt_model,
                    out_file=None,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    max_depth=6,
                    proportion=True
                )
                
                graph_medium = graphviz.Source(dot_data_medium)
                if save_files:
                    graph_medium.render('decision_tree_graphviz_MEDIUM', format='png', cleanup=True)
                    print("💾 Graphviz ORTA DETAY: 'decision_tree_graphviz_MEDIUM.png'")
                
            except Exception as e:
                print(f"⚠️ Graphviz görselleştirme hatası: {e}")
        else:
            print("\n   ℹ️ Graphviz kurulu değil - sadece matplotlib görselleştirmeleri oluşturuldu")
    
    def visualize_tree_depth_analysis(self):
        """Ağaç derinliği analizi"""
        print("\n📊 AĞAÇ DERİNLİĞİ ANALİZİ")
        print("-" * 30)
        
        dt_model = self.models['Decision Tree']
        
        # Farklı derinliklerde performans analizi
        depths = range(1, 11)
        train_scores = []
        test_scores = []
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        for depth in depths:
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            temp_model.fit(X_train, y_train)
            
            train_score = temp_model.score(X_train, y_train)
            test_score = temp_model.score(X_test, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        # Görselleştirme
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(depths, train_scores, 'o-', label='Eğitim Skoru', color='blue')
        plt.plot(depths, test_scores, 'o-', label='Test Skoru', color='red')
        plt.xlabel('Maksimum Derinlik')
        plt.ylabel('Doğruluk')
        plt.title('Ağaç Derinliği vs Performans')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Düğüm sayısı analizi
        node_counts = []
        leaf_counts = []
        
        for depth in depths:
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            temp_model.fit(X_train, y_train)
            node_counts.append(temp_model.tree_.node_count)
            leaf_counts.append(temp_model.tree_.n_leaves)
        
        plt.subplot(1, 2, 2)
        plt.plot(depths, node_counts, 'o-', label='Toplam Düğüm', color='green')
        plt.plot(depths, leaf_counts, 'o-', label='Yaprak Düğüm', color='orange')
        plt.xlabel('Maksimum Derinlik')
        plt.ylabel('Düğüm Sayısı')
        plt.title('Ağaç Derinliği vs Düğüm Sayısı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tree_depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Derinlik analizi 'tree_depth_analysis.png' olarak kaydedildi")
    
    def visualize_feature_importance_comparison(self):
        """Özellik önemlerini karşılaştır"""
        print("\n📈 ÖZELLİK ÖNEMLERİ KARŞILAŞTIRMASI")
        print("-" * 40)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Decision Tree özellik önemleri
        dt_importances = self.models['Decision Tree'].feature_importances_
        indices_dt = np.argsort(dt_importances)[::-1]
        
        axes[0].bar(range(len(dt_importances)), dt_importances[indices_dt], color='lightblue')
        axes[0].set_title('Decision Tree - Özellik Önemleri', fontweight='bold')
        axes[0].set_xlabel('Özellikler')
        axes[0].set_ylabel('Önem Skoru')
        axes[0].set_xticks(range(len(dt_importances)))
        axes[0].set_xticklabels([self.feature_names[i] for i in indices_dt], rotation=45)
        
        # Random Forest özellik önemleri
        rf_importances = self.models['Random Forest'].feature_importances_
        indices_rf = np.argsort(rf_importances)[::-1]
        
        axes[1].bar(range(len(rf_importances)), rf_importances[indices_rf], color='lightgreen')
        axes[1].set_title('Random Forest - Özellik Önemleri', fontweight='bold')
        axes[1].set_xlabel('Özellikler')
        axes[1].set_ylabel('Önem Skoru')
        axes[1].set_xticks(range(len(rf_importances)))
        axes[1].set_xticklabels([self.feature_names[i] for i in indices_rf], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detaylı karşılaştırma tablosu
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Feature_TR': [self.feature_names_tr[f] for f in self.feature_names],
            'Decision_Tree': dt_importances,
            'Random_Forest': rf_importances
        }).sort_values('Decision_Tree', ascending=False)
        
        print("\n📋 Özellik Önemleri Detaylı Karşılaştırma:")
        print(importance_df.to_string(index=False, float_format='%.4f'))
        
        return importance_df
    
    def visualize_decision_paths(self, sample_data=None):
        """Karar yollarını görselleştir"""
        print("\n🛤️ KARAR YOLLARI ANALİZİ")
        print("-" * 30)
        
        dt_model = self.models['Decision Tree']
        
        if sample_data is None:
            # Örnek veriler oluştur
            sample_data = [
                {'buying': 'low', 'maint': 'low', 'doors': '4', 'persons': 'more', 'lug_boot': 'big', 'safety': 'high'},
                {'buying': 'vhigh', 'maint': 'vhigh', 'doors': '2', 'persons': '2', 'lug_boot': 'small', 'safety': 'low'},
                {'buying': 'med', 'maint': 'med', 'doors': '3', 'persons': '4', 'lug_boot': 'med', 'safety': 'med'}
            ]
        
        from sklearn.tree import _tree
        
        def get_decision_path(model, sample):
            """Belirli bir örnek için karar yolunu al"""
            # Sample'ı encode et
            encoded_sample = []
            for feature in self.feature_names:
                value = sample[feature]
                encoded_value = self.label_encoders[feature].transform([value])[0]
                encoded_sample.append(encoded_value)
            
            # Karar yolunu al
            leaf_id = model.decision_path([encoded_sample]).toarray()[0]
            feature_names = self.feature_names
            
            path_info = []
            tree = model.tree_
            
            for node_id in range(len(leaf_id)):
                if leaf_id[node_id] == 1:
                    if tree.children_left[node_id] != _tree.TREE_LEAF:
                        feature_idx = tree.feature[node_id]
                        threshold = tree.threshold[node_id]
                        feature_name = feature_names[feature_idx]
                        
                        # Threshold'u orijinal değere çevir
                        original_values = self.label_encoders[feature_name].classes_
                        threshold_value = original_values[int(threshold)]
                        
                        path_info.append({
                            'node': node_id,
                            'feature': feature_name,
                            'threshold': threshold_value,
                            'sample_value': sample[feature_name]
                        })
            
            # Son tahmin
            prediction_encoded = model.predict([encoded_sample])[0]
            prediction = self.label_encoders['class'].inverse_transform([prediction_encoded])[0]
            
            return path_info, prediction
        
        # Her örnek için karar yolunu göster
        for i, sample in enumerate(sample_data, 1):
            print(f"\n🎯 Örnek {i}: {sample}")
            path, prediction = get_decision_path(dt_model, sample)
            print(f"📊 Tahmin: {prediction.upper()}")
            print("🛤️ Karar Yolu:")
            
            for step in path:
                print(f"   {step['feature']} = {step['sample_value']} (threshold: {step['threshold']})")
    
    def create_model_architecture_diagram(self):
        """Model mimarisi diyagramı oluştur"""
        print("\n🏗️ MODEL MİMARİSİ DİYAGRAMI")
        print("-" * 35)
        
        from sklearn.tree import _tree  # Import burada
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Mimarileri ve Özellikleri', fontsize=16, fontweight='bold')
        
        # 1. Decision Tree yapısı
        dt_info = {
            'Nodes': self.models['Decision Tree'].tree_.node_count,
            'Leaves': self.models['Decision Tree'].tree_.n_leaves,
            'Max Depth': self.models['Decision Tree'].tree_.max_depth,
            'Features': len(self.feature_names)
        }
        
        axes[0, 0].bar(dt_info.keys(), dt_info.values(), color='lightblue')
        axes[0, 0].set_title('Decision Tree - Yapısal Bilgiler', fontweight='bold')
        axes[0, 0].set_ylabel('Sayı')
        for i, (key, value) in enumerate(dt_info.items()):
            axes[0, 0].text(i, value + max(dt_info.values()) * 0.01, str(value), 
                           ha='center', fontweight='bold')
        
        # 2. Random Forest yapısı
        rf_model = self.models['Random Forest']
        tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in rf_model.estimators_]
        
        axes[0, 1].bar(range(len(tree_depths)), tree_depths, alpha=0.7, label='Max Depth')
        axes[0, 1].set_title('Random Forest - Ağaç Derinlikleri', fontweight='bold')
        axes[0, 1].set_xlabel('Ağaç #')
        axes[0, 1].set_ylabel('Derinlik')
        axes[0, 1].legend()
        
        # 3. Özellik kullanım sıklığı
        feature_usage = np.zeros(len(self.feature_names))
        dt_tree = self.models['Decision Tree'].tree_
        
        for i in range(dt_tree.node_count):
            if dt_tree.children_left[i] != _tree.TREE_LEAF:
                feature_usage[dt_tree.feature[i]] += 1
        
        axes[1, 0].pie(feature_usage, labels=self.feature_names, autopct='%1.1f%%')
        axes[1, 0].set_title('Decision Tree - Özellik Kullanım Sıklığı', fontweight='bold')
        
        # 4. Model karmaşıklığı karşılaştırması
        complexity_metrics = {
            'Decision Tree': {
                'Parametreler': dt_info['Nodes'],
                'Derinlik': dt_info['Max Depth'],
                'Yaprak': dt_info['Leaves']
            },
            'Random Forest': {
                'Parametreler': sum(tree_nodes),
                'Ortalama Derinlik': np.mean(tree_depths),
                'Ağaç Sayısı': len(rf_model.estimators_)
            }
        }
        
        metrics = ['Parametreler', 'Derinlik/Ort.Derinlik', 'Yaprak/Ağaç Sayısı']
        dt_values = [complexity_metrics['Decision Tree']['Parametreler'],
                    complexity_metrics['Decision Tree']['Derinlik'],
                    complexity_metrics['Decision Tree']['Yaprak']]
        rf_values = [complexity_metrics['Random Forest']['Parametreler'],
                    complexity_metrics['Random Forest']['Ortalama Derinlik'],
                    complexity_metrics['Random Forest']['Ağaç Sayısı']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Değerleri normalize et (görselleştirme için)
        dt_norm = [v/max(dt_values) for v in dt_values]
        rf_norm = [v/max(rf_values) for v in rf_values]
        
        axes[1, 1].bar(x - width/2, dt_norm, width, label='Decision Tree', alpha=0.8)
        axes[1, 1].bar(x + width/2, rf_norm, width, label='Random Forest', alpha=0.8)
        axes[1, 1].set_title('Model Karmaşıklığı (Normalize)', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_architecture_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Model mimarisi 'model_architecture_diagram.png' olarak kaydedildi")
    
    def export_model_summary(self):
        """Model özetini dosyaya çıkar"""
        print("\n📄 MODEL ÖZETİ OLUŞTURULUYOR")
        print("-" * 35)
        
        with open('model_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("CAR EVALUATION - MODEL YAPILARI VE ÖZETLERİ\n")
            f.write("=" * 50 + "\n\n")
            
            # Decision Tree özeti
            dt_model = self.models['Decision Tree']
            f.write("1. DECISION TREE MODELİ\n")
            f.write("-" * 25 + "\n")
            f.write(f"Toplam Düğüm Sayısı: {dt_model.tree_.node_count}\n")
            f.write(f"Yaprak Düğüm Sayısı: {dt_model.tree_.n_leaves}\n")
            f.write(f"Maksimum Derinlik: {dt_model.tree_.max_depth}\n")
            f.write(f"Özellik Sayısı: {len(self.feature_names)}\n\n")
            
            # Random Forest özeti
            rf_model = self.models['Random Forest']
            f.write("2. RANDOM FOREST MODELİ\n")
            f.write("-" * 25 + "\n")
            f.write(f"Ağaç Sayısı: {len(rf_model.estimators_)}\n")
            f.write(f"Ortalama Derinlik: {np.mean([tree.tree_.max_depth for tree in rf_model.estimators_]):.2f}\n")
            f.write(f"Toplam Düğüm: {sum([tree.tree_.node_count for tree in rf_model.estimators_])}\n\n")
            
            # Özellik önemleri
            f.write("3. ÖZELLİK ÖNEMLERİ\n")
            f.write("-" * 20 + "\n")
            f.write("Decision Tree:\n")
            for i, importance in enumerate(dt_model.feature_importances_):
                f.write(f"  {self.feature_names[i]}: {importance:.4f}\n")
            
            f.write("\nRandom Forest:\n")
            for i, importance in enumerate(rf_model.feature_importances_):
                f.write(f"  {self.feature_names[i]}: {importance:.4f}\n")
        
        print("💾 Model özeti 'model_summary_report.txt' olarak kaydedildi")
    
    def run_complete_visualization(self):
        """Tüm görselleştirmeleri çalıştır"""
        print("🎨 MODEL GÖRSELLEŞTİRME SİSTEMİ")
        print("=" * 50)
        
        # 1. Modelleri ve veriyi yükle
        self.load_models_and_data()
        
        # 2. Karar ağacı yapısını görselleştir
        self.visualize_decision_tree_structure()
        
        # 3. Ağaç derinliği analizi
        self.visualize_tree_depth_analysis()
        
        # 4. Özellik önemleri karşılaştırması
        importance_df = self.visualize_feature_importance_comparison()
        
        # 5. Karar yolları analizi
        self.visualize_decision_paths()
        
        # 6. Model mimarisi diyagramı
        self.create_model_architecture_diagram()
        
        # 7. Model özetini çıkar
        self.export_model_summary()
        
        print("\n🎉 TÜM GÖRSELLEŞTİRMELER TAMAMLANDI!")
        print("📁 Oluşturulan dosyalar:")
        files = [
            'decision_tree_FULL_DETAIL.png (TAM DETAY - Çok büyük)',
            'decision_tree_MEDIUM_DETAIL.png (6 seviye)',
            'decision_tree_SUMMARY.png (3 seviye)',
            'decision_tree_rules_FULL_DETAIL.txt (Tüm kurallar)',
            'decision_tree_rules_SUMMARY.txt (İlk 5 seviye)',
            'tree_depth_analysis.png',
            'feature_importance_comparison.png',
            'model_architecture_diagram.png',
            'model_summary_report.txt'
        ]
        
        if GRAPHVIZ_AVAILABLE:
            files.extend([
                'decision_tree_graphviz_FULL.png (TAM DETAY)',
                'decision_tree_graphviz_FULL.svg (Ölçeklenebilir)',
                'decision_tree_graphviz_FULL.pdf (Yüksek kalite)',
                'decision_tree_graphviz_MEDIUM.png (6 seviye)'
            ])
        
        print("\n✅ OLUŞTURULAN DOSYALAR:")
        for file in files:
            # Dosya adını al (açıklama varsa ayır)
            file_name = file.split(' (')[0] if ' (' in file else file
            if os.path.exists(file_name):
                print(f"   ✅ {file}")
            else:
                print(f"   ⏳ {file}")


if __name__ == "__main__":
    visualizer = ModelVisualizer()
    visualizer.run_complete_visualization()