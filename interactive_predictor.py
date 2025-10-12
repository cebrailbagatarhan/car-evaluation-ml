"""
Car Evaluation Model - İnteraktif Tahmin Sistemi
===============================================

Bu script, kullanıcıdan veri alarak araba değerlendirmesi tahminleri yapar.
Farklı veri giriş yöntemleri sunar.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import json

class CarEvaluationPredictor:
    def __init__(self):
        """Tahmin sınıfı başlatıcısı"""
        self.model = None
        self.label_encoders = {}
        self.feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        self.class_name = 'class'
        
        # Özellik değerleri
        self.feature_values = {
            'buying': ['vhigh', 'high', 'med', 'low'],
            'maint': ['vhigh', 'high', 'med', 'low'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']
        }
        
        # Türkçe açıklamalar
        self.feature_descriptions = {
            'buying': 'Satın alma fiyatı',
            'maint': 'Bakım maliyeti',
            'doors': 'Kapı sayısı',
            'persons': 'Kişi kapasitesi',
            'lug_boot': 'Bagaj boyutu',
            'safety': 'Güvenlik seviyesi'
        }
        
        # Sınıf açıklamaları
        self.class_descriptions = {
            'unacc': 'Kabul edilemez',
            'acc': 'Kabul edilebilir',
            'good': 'İyi',
            'vgood': 'Çok iyi'
        }
    
    def train_and_save_model(self):
        """Modeli eğitir ve kaydeder"""
        print("🚀 Model eğitiliyor ve kaydediliyor...")
        
        # Veri setini yükle
        df = pd.read_csv('car.data', names=self.feature_names + [self.class_name])
        
        # Label encoding
        df_encoded = df.copy()
        for feature in self.feature_names + [self.class_name]:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df[feature])
            self.label_encoders[feature] = le
        
        # Model eğitimi
        X = df_encoded[self.feature_names].values
        y = df_encoded[self.class_name].values
        
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X, y)
        
        # Modeli kaydet
        self.save_model()
        print("✅ Model başarıyla eğitildi ve kaydedildi!")
    
    def save_model(self):
        """Modeli ve encoder'ları dosyaya kaydet"""
        with open('car_evaluation_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # JSON formatında da kaydet (alternatif)
        encoders_dict = {}
        for feature, encoder in self.label_encoders.items():
            encoders_dict[feature] = {
                'classes': encoder.classes_.tolist(),
                'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
            }
        
        with open('encoders.json', 'w', encoding='utf-8') as f:
            json.dump(encoders_dict, f, ensure_ascii=False, indent=2)
    
    def load_model(self):
        """Kaydedilmiş modeli yükle"""
        try:
            with open('car_evaluation_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            print("✅ Model başarıyla yüklendi!")
            return True
        except FileNotFoundError:
            print("❌ Model dosyası bulunamadı. Önce modeli eğitin.")
            return False
    
    def predict_single(self, input_data):
        """Tek bir örnek için tahmin yap"""
        # Veriyi encode et
        encoded_data = []
        for feature in self.feature_names:
            if feature in input_data:
                value = input_data[feature]
                if value in self.label_encoders[feature].classes_:
                    encoded_value = self.label_encoders[feature].transform([value])[0]
                    encoded_data.append(encoded_value)
                else:
                    raise ValueError(f"Geçersiz değer '{value}' for {feature}. Geçerli değerler: {self.feature_values[feature]}")
            else:
                raise ValueError(f"Eksik özellik: {feature}")
        
        # Tahmin yap
        prediction_encoded = self.model.predict([encoded_data])[0]
        prediction = self.label_encoders[self.class_name].inverse_transform([prediction_encoded])[0]
        
        # Tahmin olasılıkları (eğer destekleniyorsa)
        try:
            probabilities = self.model.predict_proba([encoded_data])[0]
            class_names = self.label_encoders[self.class_name].classes_
            prob_dict = dict(zip(class_names, probabilities))
        except:
            prob_dict = None
        
        return prediction, prob_dict
    
    def interactive_input(self):
        """Kullanıcıdan interaktif veri girişi al"""
        print("\n🚗 ARABA DEĞERLENDİRME SİSTEMİ")
        print("=" * 50)
        print("Aşağıdaki özellikleri girin:")
        
        input_data = {}
        
        for feature in self.feature_names:
            print(f"\n📋 {self.feature_descriptions[feature].upper()} ({feature}):")
            for i, value in enumerate(self.feature_values[feature], 1):
                print(f"  {i}. {value}")
            
            while True:
                try:
                    choice = input(f"Seçiminizi yapın (1-{len(self.feature_values[feature])}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(self.feature_values[feature]):
                        input_data[feature] = self.feature_values[feature][choice_idx]
                        print(f"✅ Seçilen: {input_data[feature]}")
                        break
                    else:
                        print("❌ Geçersiz seçim. Tekrar deneyin.")
                except ValueError:
                    print("❌ Lütfen geçerli bir sayı girin.")
        
        return input_data
    
    def batch_input_from_file(self, filename):
        """Dosyadan toplu veri girişi"""
        try:
            # CSV dosyasından okuma
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            # JSON dosyasından okuma
            elif filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError("Desteklenen dosya formatları: .csv, .json")
            
            predictions = []
            for _, row in df.iterrows():
                input_data = row.to_dict()
                prediction, probabilities = self.predict_single(input_data)
                predictions.append({
                    'input': input_data,
                    'prediction': prediction,
                    'prediction_tr': self.class_descriptions[prediction],
                    'probabilities': probabilities
                })
            
            return predictions
        
        except Exception as e:
            print(f"❌ Dosya okuma hatası: {e}")
            return None
    
    def manual_input(self):
        """Manuel veri girişi (dictionary formatında)"""
        print("\n📝 MANUEL VERİ GİRİŞİ")
        print("=" * 40)
        print("Örnek format:")
        print("buying='high', maint='med', doors='4', persons='4', lug_boot='big', safety='high'")
        print("\nGeçerli değerler:")
        for feature, values in self.feature_values.items():
            print(f"  {feature}: {values}")
        
        input_data = {}
        for feature in self.feature_names:
            while True:
                value = input(f"{feature}: ").strip().lower()
                if value in [v.lower() for v in self.feature_values[feature]]:
                    # Doğru case'i bul
                    for correct_value in self.feature_values[feature]:
                        if value == correct_value.lower():
                            input_data[feature] = correct_value
                            break
                    break
                else:
                    print(f"❌ Geçersiz değer. Geçerli değerler: {self.feature_values[feature]}")
        
        return input_data
    
    def display_prediction(self, prediction, probabilities, input_data):
        """Tahmin sonucunu güzel bir şekilde göster"""
        print("\n" + "="*60)
        print("🔮 TAHMİN SONUCU")
        print("="*60)
        
        print("\n📊 GİRİLEN VERİLER:")
        for feature, value in input_data.items():
            print(f"  {self.feature_descriptions[feature]:.<20}: {value}")
        
        print(f"\n🎯 TAHMİN: {prediction.upper()}")
        print(f"🏷️  AÇIKLAMA: {self.class_descriptions[prediction]}")
        
        if probabilities:
            print(f"\n📈 TAHMİN OLASILIKLARI:")
            for class_name, prob in probabilities.items():
                percentage = prob * 100
                bar_length = int(percentage / 5)  # Her 5% için bir |
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {class_name:>6} ({self.class_descriptions[class_name]:.<15}): {percentage:5.1f}% {bar}")
        
        print("="*60)
    
    def create_sample_files(self):
        """Örnek dosyalar oluştur"""
        # CSV örneği
        sample_csv_data = [
            {'buying': 'high', 'maint': 'med', 'doors': '4', 'persons': '4', 'lug_boot': 'big', 'safety': 'high'},
            {'buying': 'low', 'maint': 'low', 'doors': '4', 'persons': 'more', 'lug_boot': 'big', 'safety': 'high'},
            {'buying': 'vhigh', 'maint': 'vhigh', 'doors': '2', 'persons': '2', 'lug_boot': 'small', 'safety': 'low'}
        ]
        
        df_sample = pd.DataFrame(sample_csv_data)
        df_sample.to_csv('sample_input.csv', index=False)
        
        # JSON örneği
        with open('sample_input.json', 'w', encoding='utf-8') as f:
            json.dump(sample_csv_data, f, ensure_ascii=False, indent=2)
        
        print("✅ Örnek dosyalar oluşturuldu:")
        print("  - sample_input.csv")
        print("  - sample_input.json")
    
    def run_interactive_system(self):
        """Ana interaktif sistem"""
        print("🚗 ARABA DEĞERLENDİRME TAHMİN SİSTEMİ")
        print("="*50)
        
        # Model yükleme veya eğitme
        if not self.load_model():
            print("Model eğitiliyor...")
            self.train_and_save_model()
        
        while True:
            print("\n📋 MENÜ:")
            print("1. İnteraktif veri girişi")
            print("2. Manuel veri girişi")
            print("3. Dosyadan toplu tahmin")
            print("4. Örnek dosyalar oluştur")
            print("5. Model görselleştirme")
            print("6. Gelişmiş model analizi")
            print("7. Çıkış")
            
            choice = input("\nSeçiminizi yapın (1-7): ").strip()
            
            if choice == '1':
                # İnteraktif giriş
                input_data = self.interactive_input()
                prediction, probabilities = self.predict_single(input_data)
                self.display_prediction(prediction, probabilities, input_data)
                
            elif choice == '2':
                # Manuel giriş
                input_data = self.manual_input()
                prediction, probabilities = self.predict_single(input_data)
                self.display_prediction(prediction, probabilities, input_data)
                
            elif choice == '3':
                # Dosyadan toplu tahmin
                filename = input("Dosya adını girin (örn: sample_input.csv): ").strip()
                predictions = self.batch_input_from_file(filename)
                if predictions:
                    print(f"\n📊 TOPLU TAHMİN SONUÇLARI ({len(predictions)} örnek):")
                    print("="*80)
                    for i, pred in enumerate(predictions, 1):
                        print(f"\n{i}. Örnek:")
                        self.display_prediction(pred['prediction'], pred['probabilities'], pred['input'])
                        
            elif choice == '4':
                # Örnek dosyalar oluştur
                self.create_sample_files()
                
            elif choice == '5':
                # Model görselleştirme
                self.visualize_models()
                
            elif choice == '6':
                # Gelişmiş model analizi
                self.advanced_model_analysis()
                
            elif choice == '7':
                print("👋 Görüşmek üzere!")
                break
                
            else:
                print("❌ Geçersiz seçim. Tekrar deneyin.")
    
    def advanced_model_analysis(self):
        """Gelişmiş model analizi çalıştır"""
        print("\n🚀 GELİŞMİŞ MODEL ANALİZİ")
        print("="*50)
        print("Bu işlem tüm modellerin karmaşık analizini yapacak.")
        print("- Confusion Matrix analizi")
        print("- ROC Curves ve AUC skorları")  
        print("- Learning Curves")
        print("- Feature Importance analizi")
        print("- Prediction Confidence analizi")
        print("- Model Performance Heatmap")
        print("- 3D Görselleştirmeler (Plotly)")
        print("- Kapsamlı analiz raporu")
        
        confirm = input("\nDevam etmek istiyor musunuz? (e/h): ").strip().lower()
        
        if confirm not in ['e', 'evet', 'y', 'yes']:
            print("❌ İşlem iptal edildi.")
            return
            
        try:
            print("\n⏳ Gelişmiş analiz başlatılıyor...")
            
            # Advanced model visualizer modülünü import et
            try:
                from advanced_model_visualizer import AdvancedModelVisualizer
                
                # Analizi çalıştır
                analyzer = AdvancedModelVisualizer()
                performance_df, report = analyzer.run_complete_analysis()
                
                print("\n✅ Gelişmiş analiz tamamlandı!")
                print("\n📊 OLUŞTURULAN GELIŞMIŞ GÖRSELLEŞTIRMELER:")
                print("   🖼️ comprehensive_confusion_matrices.png")
                print("   📈 roc_curves_analysis.png")
                print("   📚 learning_curves_analysis.png")
                print("   🎯 feature_importance_analysis.png")
                print("   🎲 prediction_confidence_analysis.png")
                print("   🌡️ model_performance_heatmap.png")
                print("   📋 comprehensive_model_analysis_report.txt")
                
                # Plotly kontrol et
                try:
                    import plotly
                    print("   🌐 3d_feature_space.html")
                    print("   📊 3d_model_comparison.html")
                except ImportError:
                    print("   ⚠️ 3D görselleştirmeler için Plotly gerekli")
                
                # En iyi modeli göster
                print(f"\n🏆 EN İYİ MODEL: {performance_df.loc[:, 'Accuracy'].idxmax()}")
                print(f"   Doğruluk: {performance_df.loc[:, 'Accuracy'].max():.4f}")
                
                # Raporu göster mi?
                show_report = input("\nDetaylı raporu görmek istiyor musunuz? (e/h): ").strip().lower()
                if show_report in ['e', 'evet', 'y', 'yes']:
                    print("\n" + "="*80)
                    print("KAPSAMLI ANALİZ RAPORU")
                    print("="*80)
                    # Raporun ilk 50 satırını göster
                    report_lines = report.split('\n')
                    for line in report_lines[:50]:
                        print(line)
                    if len(report_lines) > 50:
                        print("\n... (Detaylar comprehensive_model_analysis_report.txt dosyasında)")
                
            except ImportError as e:
                print(f"❌ Advanced model visualizer import edilemedi: {e}")
                print("Bu özellik için advanced_model_visualizer.py dosyası gerekli.")
                
        except Exception as e:
            print(f"❌ Gelişmiş analiz sırasında hata: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_models(self):
        """Model görselleştirme alt menüsü"""
        print("\n🎨 MODEL GÖRSELLEŞTİRME")
        print("="*40)
        print("1. Karar ağacı yapısını göster")
        print("2. Model yapılarını karşılaştır")
        print("3. Detaylı model analizi")
        print("4. Ana menüye dön")
        
        choice = input("\nSeçiminizi yapın (1-4): ").strip()
        
        if choice == '1':
            self.show_decision_tree()
        elif choice == '2':
            self.compare_model_structures()
        elif choice == '3':
            self.detailed_model_analysis()
        elif choice == '4':
            return
        else:
            print("❌ Geçersiz seçim.")
    
    def show_decision_tree(self):
        """Karar ağacı yapısını göster"""
        print("\n🌳 KARAR AĞACI YAPISI")
        print("-"*30)
        
        try:
            from model_visualizer import ModelVisualizer
            visualizer = ModelVisualizer()
            visualizer.load_models_and_data()
            visualizer.visualize_decision_tree_structure()
        except ImportError:
            print("❌ model_visualizer modülü bulunamadı.")
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")
    
    def compare_model_structures(self):
        """Model yapılarını karşılaştır"""
        print("\n📊 MODEL YAPILARI KARŞILAŞTIRMASI")
        print("-"*40)
        
        try:
            from model_visualizer import ModelVisualizer
            visualizer = ModelVisualizer()
            visualizer.load_models_and_data()
            visualizer.visualize_feature_importance_comparison()
            visualizer.create_model_architecture_diagram()
        except ImportError:
            print("❌ model_visualizer modülü bulunamadı.")
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")
    
    def detailed_model_analysis(self):
        """Detaylı model analizi"""
        print("\n🔍 DETAYLI MODEL ANALİZİ")
        print("-"*35)
        
        try:
            from model_visualizer import ModelVisualizer
            visualizer = ModelVisualizer()
            visualizer.run_complete_visualization()
        except ImportError:
            print("❌ model_visualizer modülü bulunamadı.")
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")


# API benzeri kullanım fonksiyonları
def predict_car(buying, maint, doors, persons, lug_boot, safety):
    """
    Basit API benzeri fonksiyon
    
    Parametreler:
        buying: 'vhigh', 'high', 'med', 'low'
        maint: 'vhigh', 'high', 'med', 'low'
        doors: '2', '3', '4', '5more'
        persons: '2', '4', 'more'
        lug_boot: 'small', 'med', 'big'
        safety: 'low', 'med', 'high'
    
    Dönüş:
        prediction: 'unacc', 'acc', 'good', 'vgood'
    """
    predictor = CarEvaluationPredictor()
    if predictor.load_model():
        input_data = {
            'buying': buying,
            'maint': maint,
            'doors': doors,
            'persons': persons,
            'lug_boot': lug_boot,
            'safety': safety
        }
        prediction, probabilities = predictor.predict_single(input_data)
        return prediction, probabilities
    else:
        return None, None


if __name__ == "__main__":
    # İnteraktif sistem başlat
    predictor = CarEvaluationPredictor()
    predictor.run_interactive_system()