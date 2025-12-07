import cv2
import sys
from biyoves import BiyoVes
import numpy as np

def create_dummy_image(path):
    # Create a black image with a simpler "face" that InsightFace WON'T detect
    # This is just to test import and pipeline initialization, not accuracy.
    # To test accuracy, the user needs to provide a real face image.
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.circle(img, (300, 300), 100, (200, 200, 200), -1) # Face
    cv2.circle(img, (270, 270), 10, (0, 0, 0), -1) # Eye
    cv2.circle(img, (330, 270), 10, (0, 0, 0), -1) # Eye
    cv2.imwrite(path, img)

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Kullanım: python test_migration.py <resim_yolu>")
        print("Örnek resim oluşturuluyor: dummy_face.jpg")
        image_path = "dummy_face.jpg"
        create_dummy_image(image_path)
    
    print(f"BiyoVes testi başlatılıyor: {image_path}")
    try:
        app = BiyoVes(image_path, verbose=True)
        print("BiyoVes sınıfı başlatıldı. InsightFace modelleri yükleniyor...")
        
        # We can test individual components
        img = cv2.imread(image_path)
        if img is None:
            print("Resim okunamadı.")
            return

        print("1. Oryantasyon Testi...")
        try:
            res = app.corrector.correct_image(img)
            print("   Oryantasyon sonucu (shape):", res.shape if res is not None else "None")
        except Exception as e:
            print(f"   Oryantasyon HATASI: {e}")

        print("2. Biyometrik İşlem Testi...")
        # Note: Face detection might fail on dummy image
        processed_img = None
        try:
            processed_img = app.processor.process_photo(img, "biyometrik")
            if processed_img is not None:
                print("   Biyometrik işlem başarılı (shape):", processed_img.shape)
            else:
                print("   Uyarı: Yüz bulunamadı (Beklenen durum eğer gerçek yüz değilse).")
        except Exception as e:
            print(f"   Biyometrik İşlem HATASI: {e}")

        print("3. Tam Pipeline Testi (2'li Layout)...")
        try:
            output_file = "test_result_2li.jpg"
            # Using the high-level create_image method
            app.create_image(photo_type="biyometrik", layout_type="2li", output_path=output_file)
            print(f"   Tam işlem başarılı! Sonuç kaydedildi: {output_file}")
        except Exception as e:
            print(f"   Tam Pipeline HATASI: {e}")

        print("\nTest Tamamlandı.")

    except Exception as e:
        print(f"Genel Hata: {e}")

if __name__ == "__main__":
    main()
