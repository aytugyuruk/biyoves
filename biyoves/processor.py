import cv2
import mediapipe as mp
import numpy as np

class BiometricIDGenerator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        self.DPI = 300
        self.PIXELS_PER_MM = self.DPI / 25.4

        self.PHOTO_SPECS = {
            "biyometrik": {"w": 50, "h": 60, "face_h": 30, "top_margin": 3},
            "vesikalik": {"w": 45, "h": 60, "face_h": 27, "top_margin": 3},
            "abd_vizesi": {"w": 50, "h": 50, "face_h": 25, "top_margin": 3},
            "schengen": {"w": 35, "h": 45, "face_h": 23, "top_margin": 3}
        }

    def _get_hair_top_y(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(img_rgb)
        if results.segmentation_mask is None: return None
        mask = results.segmentation_mask > 0.5
        rows, _ = np.where(mask)
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

        results_mesh = self.face_mesh.process(img_rgb)
        if not results_mesh.multi_face_landmarks:
            print("Hata: İşlenecek yüz bulunamadı.")
            return None

        landmarks = results_mesh.multi_face_landmarks[0].landmark
        forehead_y = int(landmarks[10].y * h_orig)
        chin_y = int(landmarks[152].y * h_orig)
        right_ear_x = int(landmarks[234].x * w_orig)
        left_ear_x = int(landmarks[454].x * w_orig)
        
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