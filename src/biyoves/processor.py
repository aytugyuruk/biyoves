import cv2
import numpy as np
import logging
import os
from .face_utils import SCRFD, Landmark106, Face

logger = logging.getLogger(__name__)

class BiometricIDGenerator:
    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        det_path = os.path.join(package_dir, "det_500m.onnx")
        lm_path = os.path.join(package_dir, "2d106det.onnx")

        try:
            if not os.path.exists(det_path): raise FileNotFoundError(f"Model not found: {det_path}")
            if not os.path.exists(lm_path): raise FileNotFoundError(f"Model not found: {lm_path}")

            self.detector = SCRFD(det_path)
            self.detector.prepare(0)
            
            self.landmarker = Landmark106(lm_path)
            
        except Exception as e:
            logger.error(f"Model başlatma hatası: {e}")
            raise

        self.DPI = 300
        self.PIXELS_PER_MM = self.DPI / 25.4

        self.PHOTO_SPECS = {
            "biyometrik": {"w": 50, "h": 60, "face_h": 34, "top_margin": 5}, # ICAO: Face 32-36mm. Target 34.
            "vesikalik": {"w": 45, "h": 60, "face_h": 30, "top_margin": 5},
            "abd_vizesi": {"w": 50, "h": 50, "face_h": 30, "top_margin": 5},
            "schengen": {"w": 35, "h": 45, "face_h": 28, "top_margin": 4}
        }

    def _get_landmarks(self, face):
        """Returns key landmarks: left_eye, right_eye, chin, nose_tip"""
        if face.landmark_2d_106 is not None:
            lms = face.landmark_2d_106
            # 106 mapping (approximate for standard InsightFace models):
            # 104: Left Eye Center (approx)
            # 105: Right Eye Center (approx)
            # 16: Chin
            # 46: Nose tip (approx)
            # 72-73: Left Eye
            # 75-76: Right Eye
            # Let's use robust indices or averages
            
            # Using 106 points indices:
            # Chin is usually index 16
            chin = lms[16]
            
            # Eyes: 33-42 (Eyebrow L), 43-52 (R), 53-60 (Eye L), 61-68 (Eye R) ?
            # Different models have different indices. 
            # Safest fallback is face.kps which is always 5 points.
            # But we want 106 for better chin?
            # Let's stick to kps for Eyes (very stable). Use 106 for Chin if available.
            return face.kps[0], face.kps[1], lms[16], face.kps[2] # LE, RE, Chin, Nose
        
        # Fallback to 5 KPS
        # Estimate chin from nose-mouth distance
        # Nose: 2, Left Mouth: 3, Right Mouth: 4
        nose = face.kps[2]
        mouth_center = (face.kps[3] + face.kps[4]) / 2
        nose_mouth_dist = np.linalg.norm(nose - mouth_center)
        estimated_chin = mouth_center + (mouth_center - nose) * 0.8 # approx
        
        return face.kps[0], face.kps[1], estimated_chin, face.kps[2]

    def _estimate_hair_top(self, left_eye, right_eye, chin):
        """Estimates top of skull/hair based on eye and chin positions"""
        eye_center = (left_eye + right_eye) / 2
        chin_y = chin[1]
        eye_y = eye_center[1]
        
        face_bottom_half = chin_y - eye_y
        # Typical face: Eyes are at mid-height of head (skull).
        # Hair adds volume.
        # Estimate top = eye - face_bottom_half * factor
        # Factor 1.0 = Top of skull. Factor 1.3-1.4 = Top of hair.
        
        return eye_y - (face_bottom_half * 1.5)

    def process_photo(self, image_input, photo_type="biyometrik"):
        if photo_type not in self.PHOTO_SPECS:
            logger.error(f"Hata: Geçersiz tür '{photo_type}'")
            return None
            
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
        else:
            original_image = image_input

        if original_image is None: return None

        # 1. Detect faces
        dets, kpss = self.detector.detect(original_image, max_num=0)
        if kpss is None or len(kpss) == 0:
            logger.warning("Hata: İşlenecek yüz bulunamadı.")
            return None
            
        # Pick largest face
        # dets: [x1, y1, x2, y2, score]
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        largest_idx = np.argmax(areas)
        
        bbox = dets[largest_idx][:4]
        kps = kpss[largest_idx]
        
        # Get 106 landmarks
        lms106 = self.landmarker.get(original_image, bbox)
        
        # Create Face object to be compatible with existing logic
        face = Face(bbox=bbox, kps=kps, lms106=lms106, det_score=dets[largest_idx][4])
        
        # 1. Alignment (Rotation)
        left_eye, right_eye, chin, nose = self._get_landmarks(face)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image to make eyes horizontal
        h, w = original_image.shape[:2]
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(original_image, M_rot, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))
        
        # 2. Re-detect on rotated image for precise cropping
        # Only needed if rotation was significant (>5 deg)
        if abs(angle) > 5:
            # Re-detect on rotated image
            dets_rot, kpss_rot = self.detector.detect(rotated_img)
            if kpss_rot is not None and len(kpss_rot) > 0:
                 areas = (dets_rot[:, 2] - dets_rot[:, 0]) * (dets_rot[:, 3] - dets_rot[:, 1])
                 largest_idx = np.argmax(areas)
                 bbox = dets_rot[largest_idx][:4]
                 lms106 = self.landmarker.get(rotated_img, bbox)
                 face = Face(bbox=bbox, kps=kpss_rot[largest_idx], lms106=lms106)
                 
                 left_eye, right_eye, chin, nose = self._get_landmarks(face)

        # 3. Scaling & Cropping
        spec = self.PHOTO_SPECS[photo_type]
        
        hair_top_y = self._estimate_hair_top(left_eye, right_eye, chin)
        face_height_px = abs(chin[1] - hair_top_y)
        
        target_face_h_px = spec['face_h'] * self.PIXELS_PER_MM
        scale = target_face_h_px / face_height_px
        
        # Target canvas size
        target_w = int(spec['w'] * self.PIXELS_PER_MM)
        target_h = int(spec['h'] * self.PIXELS_PER_MM)
        target_top_margin_px = int(spec['top_margin'] * self.PIXELS_PER_MM)
        
        # We need to map the "hair top" on original/rotated image to "target_top_margin" on canvas
        # And center horizontally based on nose/eyes center
        
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        
        # New coordinates after scaling
        new_face_center_x = face_center_x * scale
        new_hair_top_y = hair_top_y * scale
        
        # Shifts
        shift_x = (target_w / 2) - new_face_center_x
        shift_y = target_top_margin_px - new_hair_top_y
        
        M_scale_trans = np.float32([
            [scale, 0, shift_x],
            [0, scale, shift_y]
        ])
        
        final_canvas = cv2.warpAffine(rotated_img, M_scale_trans, (target_w, target_h), 
                                      flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))
        
        return final_canvas