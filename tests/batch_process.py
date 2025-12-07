import os
import sys
import glob
# Ensure we map from source if running from root
sys.path.append("src")
from biyoves import BiyoVes

def process_batch():
    source_dir = "resimler"
    output_dir = "sonuclar"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    supported_ext = [".png", ".jpg", ".jpeg"]
    files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in supported_ext]
    
    print(f"Toplam {len(files)} resim bulundu. İşleniyor...")
    
    for filename in files:
        file_path = os.path.join(source_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n--- İşleniyor: {filename} ---")
        try:
            app = BiyoVes(file_path, verbose=False)
            
            # 1. Biyometrik 4'lü
            out_bio = os.path.join(output_dir, f"{base_name}_biyometrik_4lu.jpg")
            print(f"   > Biyometrik 4'lü oluşturuluyor...")
            app.create_image("biyometrik", "4lu", out_bio)
            
            # 2. Vesikalık 4'lü
            # Note: We can reuse the same app instance or reload if needed. 
            # Ideally reusing is faster as models are loaded.
            # However create_image implementation re-runs pipeline from scratch usually.
            
            out_ves = os.path.join(output_dir, f"{base_name}_vesikalik_4lu.jpg")
            print(f"   > Vesikalık 4'lü oluşturuluyor...")
            app.create_image("vesikalik", "4lu", out_ves)
            
            print(f"   [OK] Tamamlandı.")
            
        except Exception as e:
            print(f"   [HATA] {filename} işlenirken hata oluştu: {e}")

if __name__ == "__main__":
    process_batch()
