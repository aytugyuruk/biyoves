import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

class BiometricIDGenerator:
    def __init__(self):
        if FaceAnalysis is None:
            raise ImportError(
                "InsightFace kutuphanesi yuklenmemis. \n"
                "Kurulum: pip install insightface"
            )
        # InsightFace modelini başlat - landmarks ve segmentasyon için
        self.face_app = FaceAnalysis(name="buffalo_l", providers=['CPUProvider'])
        self.face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

        self.DPI = 300
        self.PIXELS_PER_MM = self.DPI / 25.4

        self.PHOTO_SPECS = {
            "biyometrik": {"w": 50, "h": 60, "face_h": 30, "top_margin": 3},
            "vesikalik": {"w": 45, "h": 60, "face_h": 27, "top_margin": 3},
            "abd_vizesi": {"w": 50, "h": 50, "face_h": 25, "top_margin": 3},
            "schengen": {"w": 35, "h": 45, "face_h": 23, "top_margin": 3}
        }

    def _get_hair_top_y(self, image):
        """Basit ön işleme yaparak saç başı pozisyonunu tahmin et"""
        # Gri tona dönüştür ve köprü bulma
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold ile cilt/vücut ayrımı yap
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # Saçlara değiştirilmek için ön tarafı kes
        h = image.shape[0]
        binary[h//2:, :] = 0
        rows, _ = np.where(binary)
        return np.min(rows) if len(rows) > 0 else 0

    def process_photo(self, image_input, photo_type="biyometrik"):
        if photo_type not in self.PHOTO_SPECS:
            print(f"Hata: Geçersiz tür '{photo_type}'")
            return None
            
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
        else:
            original_image = image_input

        if original_image is None: return None

        spec = self.PHOTO_SPECS[photo_type]
        h_orig, w_orig, _ = original_image.shape
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # InsightFace ile yüz tespiti ve landmarks al
        faces = self.face_app.get(img_rgb)
        if not faces:
            print("Hata: İşlenecek yüz bulunamadı.")
            return None

        # İlk yüzü al
        face = faces[0]
        landmarks = face.landmark_3d_68  # 68 point landmarks
        
        # Önemli noktaları al (10 = alın, 8 = çene, 0 = sol yüz, 16 = sağ yüz)
        forehead_y = int(landmarks[27, 1])  # Burun başı
        chin_y = int(landmarks[8, 1])  # Çene ucu
        right_ear_x = int(landmarks[16, 0])  # Sağ yüz knarı
        left_ear_x = int(landmarks[0, 0])  # Sol yüz knarı
        
        face_center_x = (right_ear_x + left_ear_x) // 2
        hair_top_y = self._get_hair_top_y(original_image)
        if hair_top_y is None:
            hair_top_y = max(0, forehead_y - int((chin_y - forehead_y) * 0.5))

        target_canvas_w = int(spec['w'] * self.PIXELS_PER_MM)
        target_canvas_h = int(spec['h'] * self.PIXELS_PER_MM)
        target_face_h = spec['face_h'] * self.PIXELS_PER_MM
        target_top_margin = spec['top_margin'] * self.PIXELS_PER_MM

        current_face_h = abs(chin_y - forehead_y)
        scale_factor = target_face_h / current_face_h
        
        new_w, new_h = int(w_orig * scale_factor), int(h_orig * scale_factor)
        resized_img = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        new_hair_top_y = int(hair_top_y * scale_factor)
        new_face_center_x = int(face_center_x * scale_factor)

        shift_x = (target_canvas_w / 2) - new_face_center_x
        shift_y = target_top_margin - new_hair_top_y

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Maksimum kalite için LANCZOS4 interpolation kullan
        final_canvas = cv2.warpAffine(resized_img, M, (target_canvas_w, target_canvas_h), 
                                      flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))
        
        return final_canvas