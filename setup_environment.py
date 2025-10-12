# Python Environment Setup and Package Installation
# Bu dosya projeyi çalıştırmak için gerekli tüm Python paketlerini yükler

import subprocess
import sys
import os

def install_package(package):
    """Paketi yükle"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} başarıyla yüklendi")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} yüklenirken hata oluştu")
        return False

def main():
    """Ana fonksiyon - tüm gerekli paketleri yükle"""
    print("🚀 Python Environment Setup başlıyor...")
    print("=" * 50)
    
    # Gerekli paketler listesi
    required_packages = [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib",
        "plotly",
        "flask",
        "graphviz"
    ]
    
    installed_count = 0
    failed_packages = []
    
    for package in required_packages:
        print(f"\n📦 {package} yükleniyor...")
        if install_package(package):
            installed_count += 1
        else:
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    print("📊 YÜK İŞLEMİ ÖZET:")
    print(f"✅ Başarılı: {installed_count}/{len(required_packages)}")
    
    if failed_packages:
        print(f"❌ Başarısız: {failed_packages}")
    else:
        print("🎉 TÜM PAKETLER BAŞARILI ŞEKİLDE YÜKLENDİ!")
    
    # Python sürümünü göster
    print(f"\n🐍 Python Sürümü: {sys.version}")
    print(f"📂 Python Yolu: {sys.executable}")
    
    # Paket sürümlerini kontrol et
    print("\n📋 YÜKLENMİŞ PAKET SÜRÜMLER:")
    print("-" * 30)
    
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except ImportError:
        print("pandas: YÜKLENMEMİŞ")
    
    try:
        import numpy as np
        print(f"numpy: {np.__version__}")
    except ImportError:
        print("numpy: YÜKLENMEMİŞ")
    
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn: YÜKLENMEMİŞ")
    
    try:
        import matplotlib
        print(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("matplotlib: YÜKLENMEMİŞ")
    
    try:
        import plotly
        print(f"plotly: {plotly.__version__}")
    except ImportError:
        print("plotly: YÜKLENMEMİŞ")
    
    try:
        import flask
        print(f"flask: {flask.__version__}")
    except ImportError:
        print("flask: YÜKLENMEMİŞ")

if __name__ == "__main__":
    main()