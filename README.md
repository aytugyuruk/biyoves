# BiyoVes 📸

Yapay Zeka destekli Biyometrik, Vesikalık ve Vize fotoğrafları oluşturma aracı.

## Kurulum

```bash
pip install biyoves
```

Veya kaynak koddan:

```bash
git clone https://github.com/aytugyuruk/biyoves.git
cd biyoves
pip install -e .
```

## Hızlı Başlangıç

### Yöntem 1: Sınıf Kullanımı (Önerilen)

```python
from biyoves import BiyoVes

# Fotoğraf yolunu belirt
img = BiyoVes("foto.jpg")

# Vesikalık fotoğraf oluştur (2li layout)
vesikalik = img.create_image("vesikalik", "2li", "sonuc_vesikalik.jpg")

# Biyometrik fotoğraf oluştur (4lu layout)
biyometrik = img.create_image("biyometrik", "4lu", "sonuc_biyometrik.jpg")

# ABD vizesi için
abd_vizesi = img.create_image("abd_vizesi", "2li", "sonuc_abd.jpg")

# Schengen vizesi için
schengen_vizesi = img.create_image("schengen", "4lu", "sonuc_schengen.jpg")
```

### Yöntem 2: Fonksiyon Kullanımı

```python
from biyoves import create_image

# Tek satırda işlem
vesikalik = create_image("foto.jpg", "vesikalik", "2li", "sonuc.jpg")
```

## Fotoğraf Tipleri

- `"biyometrik"` - Standart biyometrik fotoğraf (50x60mm)
- `"vesikalik"` - Vesikalık fotoğraf (45x60mm)
- `"abd_vizesi"` - ABD vizesi için (50x50mm)
- `"schengen"` - Schengen vizesi için (35x45mm)

## Layout Tipleri

- `"2li"` - 2 fotoğraf alt alta (2x1)
- `"4lu"` - 4 fotoğraf (2x2)

## Özellikler

✅ Yapay Zeka ile otomatik arkaplan kaldırma
✅ Yüz açısını otomatik düzeltme
✅ Standart boyutlara göre otomatik kırpma
✅ Baskı şablonları (2li/4lu)
✅ Kesim çizgileri ile hazır baskı dosyası

## Örnek Kullanım

```python
from biyoves import BiyoVes

# Fotoğrafı yükle
img = BiyoVes("insan.jpg")

# Farklı formatlarda kaydet
img.create_image("vesikalik", "2li", "vesikalik_2li.jpg")
img.create_image("vesikalik", "4lu", "vesikalik_4lu.jpg")
img.create_image("biyometrik", "2li", "biyometrik_2li.jpg")
img.create_image("abd_vizesi", "4lu", "abd_4lu.jpg")
```

## Gereksinimler

- Python >= 3.7
- OpenCV
- NumPy

- ONNX Runtime

## Lisans

MIT License