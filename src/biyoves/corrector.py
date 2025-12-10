import cv2
import numpy as np
import logging
import os
from .face_utils import SCRFD, Face

logger = logging.getLogger(__name__)

class FaceOrientationCorrector:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Initialize SCRFD model
        package_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(package_dir, "models", "det_500m.onnx")
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        self.detector = SCRFD(model_path)
        self.detector.prepare(0)

    def _rotate_image(self, image, angle):
        """
        Rotates image ACW by angle degrees (0, 90, 180, 270).
        """
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    def _calculate_required_rotation(self, kps):
        """
        Calculates required rotation angle (0, 90, 180, 270) to make face upright
        based on eye keypoints.
        InsightFace KPS: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        left_eye = kps[0]
        right_eye = kps[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle_deg = np.degrees(np.arctan2(dy, dx))
        
        # Ideal upright: 0 degrees (left eye at left, right eye at right)
        # We need to rotate the IMAGE opposite to the face angle to fix it.
        
        if -45 < angle_deg <= 45:
            return 0
        elif 45 < angle_deg <= 135:
            # Face is approx 90 deg (Clockwise). Eyes are vertical.
            # We need to rotate image 90 deg Counter-Clockwise to fix.
            return 90
        elif angle_deg > 135 or angle_deg <= -135:
            # Face is approx 180 deg (Upside down).
            return 180
        elif -135 < angle_deg <= -45:
            # Face is approx -90 (270) deg.
            # We need to rotate image 90 deg Clockwise (270 CCW) to fix.
            return 270
            
        return 0

    def correct_image(self, image_input):
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
            if original_image is None:
                logger.error(f"Resim okunamadı -> {image_input}")
                return None
        else:
            original_image = image_input

        if original_image is None: return None

        # Check all 4 orientations
        rotations = [0, 90, 180, 270]
        best_score = -1.0
        best_angle = 0
        best_img = original_image
        
        for angle in rotations:
            # Rotate image
            current_img = self._rotate_image(original_image, angle)
            
            # Detect face
            dets, kpss = self.detector.detect(current_img, max_num=1)
            
            if dets is not None and len(dets) > 0:
                # Get score of the best face
                score = dets[0][4]
                
                # Check if face is upright relative to this rotation
                # Calculate internal rotation based on eyes
                kps = kpss[0]
                left_eye = kps[0]
                right_eye = kps[1]
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                internal_angle = np.degrees(np.arctan2(dy, dx))
                
                # If detector says face is rotated > 45 degrees, score penalty
                # We want the face to be upright (eyes horizontal)
                if abs(internal_angle) > 30: # 30 degree tolerance
                    score *= 0.5 # Penalty for non-upright faces
                
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_img = current_img

        if self.verbose:
            if best_score > 0:
                logger.info(f"En iyi açı: {best_angle}° (Skor: {best_score:.4f})")
            else:
                logger.warning("Hiçbir açıda yüz bulunamadı.")
                
        return best_img