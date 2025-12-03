import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

class FaceOrientationCorrector:
    def __init__(self, verbose=False):
        if FaceAnalysis is None:
            raise ImportError(
                "InsightFace kutuphanesi yuklenmemis. \n"
                "Kurulum: pip install insightface"
            )
        self.verbose = verbose
        # InsightFace modelini başlat - yüz tespiti için
        self.face_app = FaceAnalysis(name="buffalo_l", providers=['CPUProvider'])
        self.face_app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))

    def _rotate_image(self, image, angle):
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _analyze_face(self, face, image_shape):
        """InsightFace face landmark'larını analiz et"""
        landmarks = face.landmark_3d_68
        h, w = image_shape[:2]
        
        # 36, 39 = sağ göz (0-indexed), 42, 45 = sol göz, 30 = burun, 48, 57 = ağız
        right_eye_y = landmarks[36, 1]
        left_eye_y = landmarks[42, 1]
        nose_y = landmarks[30, 1]
        mouth_y = landmarks[57, 1]

        eyes_above_nose = (right_eye_y < nose_y) and (left_eye_y < nose_y)
        nose_above_mouth = nose_y < mouth_y
        eyes_level = abs(right_eye_y - left_eye_y) < (h * 0.15)
        
        # Güven puanı (InsightFace'te bbox ve landmark confidence'ı var)
        score = face.det_score if hasattr(face, 'det_score') else 1.0
        
        return (eyes_above_nose and nose_above_mouth and eyes_level), score

    def correct_image(self, image_input):
        # Dosya yolu gelirse oku
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
        else:
            original_image = image_input

        if original_image is None: return None

        rotation_checks = [0, 90, 180, 270]
        candidates = []

        for angle in rotation_checks:
            current_img = self._rotate_image(original_image, angle)
            img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            
            # InsightFace ile yüz tespiti yap
            faces = self.face_app.get(img_rgb)
            
            if faces:
                # En yüksek güven puanına sahip yüzü al
                best_face = max(faces, key=lambda f: f.det_score if hasattr(f, 'det_score') else 0)
                is_valid, score = self._analyze_face(best_face, current_img.shape)
                if is_valid:
                    candidates.append((score, angle, current_img))

        if not candidates:
            if self.verbose:
                print("Yüz yönü tespit edilemedi, orijinal kullanılıyor.")
            return original_image
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_angle, best_img = candidates[0]
        
        if best_angle != 0 and self.verbose:
            print(f"Resim {best_angle}° döndürüldü")
        
        return best_img