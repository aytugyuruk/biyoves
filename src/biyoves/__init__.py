import cv2
from pathlib import Path

from .remove_bg import BackgroundRemover
from .corrector import FaceOrientationCorrector
from .processor import BiometricIDGenerator
from .layout import PrintLayoutGenerator
import logging

logger = logging.getLogger(__name__)




class BiyoVes:
    
    def __init__(self, image_path=None, verbose=True):

        self.verbose = verbose
        self.image_path = image_path
        
        # Modelleri yükle
        # Model dosyasının yolunu bul (paket içinde)
        package_dir = Path(__file__).parent
        model_path = package_dir / "models" / "modnet.onnx"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        self.bg_remover = BackgroundRemover(str(model_path))
        self.corrector = FaceOrientationCorrector(verbose=self.verbose)
        self.processor = BiometricIDGenerator()
        self.layout_gen = PrintLayoutGenerator()
    
    def create_image(self, photo_type="biyometrik", layout_type="2li", output_path=None):
    
        if self.image_path is None:
            raise ValueError("Fotoğraf yolu belirtilmedi. BiyoVes('foto.jpg') şeklinde başlatın.")
        
        # 1. Resmi Oku
        original_img = cv2.imread(self.image_path)
        if original_img is None:
            raise FileNotFoundError(f"Giriş resmi bulunamadı: {self.image_path}")
        
        # 2. Arkaplan Temizleme
        bg_removed_img = self.bg_remover.process(original_img)
        if bg_removed_img is None:
            raise RuntimeError("Arkaplan silme başarısız.")
        
        # 3. Yüz Yönü Düzeltme
        corrected_img = self.corrector.correct_image(bg_removed_img)
        if corrected_img is None:
            raise RuntimeError("Oryantasyon düzeltme hatası.")
        
        # 4. Biyometrik İşleme (Crop & Resize)
        processed_img = self.processor.process_photo(corrected_img, photo_type=photo_type)
        if processed_img is None:
            raise RuntimeError("Yüz bulunamadı veya işlenemedi.")
        
        # 5. Baskı Şablonu (Layout)
        final_layout = self.layout_gen.generate_layout(processed_img, layout_type=layout_type)
        
        if final_layout is None:
            raise RuntimeError("Layout oluşturulamadı.")
        
        # 6. Kaydetme (eğer output_path belirtilmişse)
        if output_path:
            # Dosya uzantısına göre kalite parametreleri - %100 kalite
            output_lower = output_path.lower()
            if output_lower.endswith('.jpg') or output_lower.endswith('.jpeg'):
                # JPEG için maksimum kalite (100 = kayıpsız)
                cv2.imwrite(output_path, final_layout, [cv2.IMWRITE_JPEG_QUALITY, 100])
            elif output_lower.endswith('.png'):
                # PNG için kayıpsız (compression 0 = en yüksek kalite)
                cv2.imwrite(output_path, final_layout, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                # Diğer formatlar için varsayılan
                cv2.imwrite(output_path, final_layout)
            
            if self.verbose:
                logger.info(f"İşlem tamamlandı: {output_path}")
        
        return final_layout
    
    def set_image(self, image_path):
        """Fotoğraf yolunu değiştir."""
        self.image_path = image_path


# Kolay kullanım için fonksiyon API'si
def create_image(image_path, photo_type="biyometrik", layout_type="2li", output_path=None, verbose=True):
    biyoves = BiyoVes(image_path, verbose=verbose)
    return biyoves.create_image(photo_type, layout_type, output_path)


__version__ = "1.0.0"
__all__ = ["BiyoVes", "create_image"]

