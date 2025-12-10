
import os
import sys
import cv2

# Add src to path to ensure we can import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from biyoves.processor import BiometricIDGenerator
from biyoves.layout import PrintLayoutGenerator

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "example.png")
    output_path = os.path.join(base_dir, "test_result_4lu_biyometrik.jpg")
    
    # Check input
    if not os.path.exists(input_path):
        print(f"Hata: Örnek dosya bulunamadı: {input_path}")
        # Try to find any png/jpg in current dir as fallback
        possible_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if possible_files:
            input_path = os.path.join(base_dir, possible_files[0])
            print(f"Alternatif dosya kullanılıyor: {input_path}")
        else:
            print("İşlenecek dosya bulunamadı.")
            return

    print("Generator başlatılıyor...")
    processor = BiometricIDGenerator()
    layout_gen = PrintLayoutGenerator()

    try:
        print(f"Biyometrik fotoğraf işleniyor: {input_path}")
        # 1. Fotoğrafı biyometrik standartlara göre işle (Arkaplan temizle, kırp, vs.)
        # Default photo_type="biyometrik" (50x60mm)
        processed_img = processor.process_photo(input_path, photo_type="biyometrik")
        
        if processed_img is None:
            print("Fotoğraf işlenemedi.")
            return

        # 2. 4'lü şablona yerleştir
        print("4'lü şablon oluşturuluyor...")
        final_layout = layout_gen.generate_layout(processed_img, layout_type="4lu")
        
        if final_layout is None:
            print("Şablon oluşturulamadı.")
            return

        # Kaydet
        cv2.imwrite(output_path, final_layout)
        print(f"Başarılı! Sonuç kaydedildi: {output_path}")
        
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
