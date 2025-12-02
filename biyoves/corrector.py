import cv2
import mediapipe as mp
import numpy as np

class FaceOrientationCorrector:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.4, model_selection=1)

    def _rotate_image(self, image, angle):
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _analyze_face(self, detection):
        score = detection.score[0]
        kp = detection.location_data.relative_keypoints
        right_eye, left_eye, nose, mouth = kp[0], kp[1], kp[2], kp[3]

        eyes_above_nose = (right_eye.y < nose.y) and (left_eye.y < nose.y)
        nose_above_mouth = nose.y < mouth.y
        eyes_level = abs(right_eye.y - left_eye.y) < 0.15 
        
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
            results = self.face_detection.process(img_rgb)
            
            if results.detections:
                best_detection = max(results.detections, key=lambda d: d.score[0])
                is_valid, score = self._analyze_face(best_detection)
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