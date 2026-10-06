import cv2
import numpy as np
from pathlib import Path

src = cv2.imread("outputs/asset_test/girl_character.jpg")
src = cv2.resize(src, (520, 522))

out = Path("assets/scenes")
out.mkdir(parents=True, exist_ok=True)

styles = [
    ((18,18,28),(55,55,70),99,250),
    ((12,20,38),(35,55,90),99,260),
    ((25,15,18),(80,45,40),99,230),
    ((10,10,12),(45,45,45),99,360),
    ((18,15,10),(75,55,35),99,350),
    ((35,25,12),(150,95,45),99,380),
]

for i, (c1, c2, y, x) in enumerate(styles, 1):
    bg1 = np.full((720,1280,3), c1, dtype=np.uint8)
    bg2 = np.full((720,1280,3), c2, dtype=np.uint8)
    bg = cv2.addWeighted(bg1, 0.55, bg2, 0.45, 0)
    bg[y:y+522, x:x+520] = src
    cv2.imwrite(str(out / f"scene_{i:02d}.png"), bg)

print("SCENE_ASSETS_READY")
