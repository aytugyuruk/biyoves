import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PrintLayoutGenerator:
    def __init__(self):
        self.DPI = 300
        self.PIXELS_PER_MM = self.DPI / 25.4
        self.LAYOUTS = {
            "2li": {"w_mm": 50, "h_mm": 150, "rows": 2, "cols": 1},
            "4lu": {"w_mm": 100, "h_mm": 150, "rows": 2, "cols": 2}
        }

    def generate_layout(self, image_input, layout_type="2li"):
        if layout_type not in self.LAYOUTS:
            logger.error(f"Hata: Geçersiz şablon '{layout_type}'")
            return None

        # Girdi dosya yolu mu yoksa görüntü matrisi mi?
        if isinstance(image_input, str):
            input_img = cv2.imread(image_input)
        else:
            input_img = image_input

        if input_img is None: return None

        layout = self.LAYOUTS[layout_type]
        img_h, img_w, _ = input_img.shape

        # Canvas (Tuval) Boyutları
        canvas_w = int(layout["w_mm"] * self.PIXELS_PER_MM)
        canvas_h = int(layout["h_mm"] * self.PIXELS_PER_MM)
        
        # Beyaz zemin oluştur
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        rows, cols = layout["rows"], layout["cols"]
        cell_w, cell_h = canvas_w // cols, canvas_h // rows

        # --- YERLEŞTİRME VE KONTÜR ---
        # Kesim kolaylığı için açık gri renk
        contour_color = (180, 180, 180) 
        contour_thickness = 2

        for r in range(rows):
            for c in range(cols):
                cx, cy = (c * cell_w) + (cell_w // 2), (r * cell_h) + (cell_h // 2)
                start_x, start_y = cx - (img_w // 2), cy - (img_h // 2)
                end_x, end_y = start_x + img_w, start_y + img_h

                y1, y2 = max(0, start_y), min(canvas_h, end_y)
                x1, x2 = max(0, start_x), min(canvas_w, end_x)
                
                img_y1, img_y2 = 0, y2 - y1
                img_x1, img_x2 = 0, x2 - x1

                if y2 > y1 and x2 > x1:
                    # 1. Resmi yapıştır
                    canvas[y1:y2, x1:x2] = input_img[img_y1:img_y2, img_x1:img_x2]
                    
                    # 2. Etrafına KONTÜR (Çerçeve) çiz
                    # cv2.rectangle(img, start_point, end_point, color, thickness)
                    # Koordinatların tam resim sınırına gelmesi için x2-1 ve y2-1 kullanıyoruz
                    cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), contour_color, contour_thickness)

        # --- GENEL KESİM ÇİZGİLERİ (Kılavuzlar) ---
        # Sayfayı ortadan bölen daha kalın siyah çizgiler
        line_color, thickness = (0, 0, 0), 2
        
        # Dikey Ayıraçlar
        for c in range(1, cols):
            x = c * cell_w
            cv2.line(canvas, (x, 0), (x, canvas_h), line_color, thickness)
            
        # Yatay Ayıraçlar
        for r in range(1, rows):
            y = r * cell_h
            cv2.line(canvas, (0, y), (canvas_w, y), line_color, thickness)

        return canvas