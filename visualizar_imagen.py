#!/usr/bin/env python3
import cv2
import numpy as np
import insightface

# Imagen específica
IMG_PATH = "/home/capistran/Documents/download_gs_path/streamax_asset_todos/186717_1766332029000_4.jpeg"

# Inicializar con RetinaFace
app = insightface.app.FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0)

# Cargar imagen
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"❌ No se pudo cargar: {IMG_PATH}")
    exit(1)

h, w, _ = img.shape
print(f"📐 Imagen: {IMG_PATH}")
print(f"📐 Dimensiones: {w}x{h}\n")

# Detectar rostros
faces = app.get(img)
print(f"🔍 Rostros detectados: {len(faces)}\n")

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=15):
    """Dibuja un rectángulo con esquinas redondeadas"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Asegurar coordenadas correctas
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Rectángulo principal
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Esquinas redondeadas
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_corner_brackets(img, pt1, pt2, color, thickness=3, bracket_size=20):
    """Dibuja marcas en las esquinas del bbox (aesthetic)"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Esquina superior izquierda
    cv2.line(img, (x1, y1), (x1 + bracket_size, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + bracket_size), color, thickness)
    
    # Esquina superior derecha
    cv2.line(img, (x2, y1), (x2 - bracket_size, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + bracket_size), color, thickness)
    
    # Esquina inferior izquierda
    cv2.line(img, (x1, y2), (x1 + bracket_size, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - bracket_size), color, thickness)
    
    # Esquina inferior derecha
    cv2.line(img, (x2, y2), (x2 - bracket_size, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - bracket_size), color, thickness)

for i, face in enumerate(faces):
    print(f"👤 Rostro {i+1}:")
    
    # Bounding box del rostro
    bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    
    print(f"  Bbox: ({x1}, {y1}) - ({x2}, {y2})")
    
    # 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    kps = face.kps
    
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]
    
    print(f"  Ojo izquierdo:     ({left_eye[0]:.2f}, {left_eye[1]:.2f})")
    print(f"  Ojo derecho:       ({right_eye[0]:.2f}, {right_eye[1]:.2f})")
    print(f"  Nariz:             ({nose[0]:.2f}, {nose[1]:.2f})")
    print(f"  Boca izq:          ({left_mouth[0]:.2f}, {left_mouth[1]:.2f})")
    print(f"  Boca der:          ({right_mouth[0]:.2f}, {right_mouth[1]:.2f})")
    
    # Calcular IPD
    ipd = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
    print(f"\n  📏 IPD: {ipd:.2f} píxeles\n")
    
    # Color elegante para bbox
    bbox_color = (30, 200, 200)  # Cian
    landmark_color = (0, 255, 0)  # Verde para landmarks
    
    # Dibujar bbox con esquinas redondeadas
    draw_rounded_rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=2, radius=15)
    
    # Dibujar marcas en esquinas (aesthetic)
    draw_corner_brackets(img, (x1, y1), (x2, y2), (255, 200, 0), thickness=3, bracket_size=25)
    
    # Dimensiones del bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Texto de dimensiones en esquina superior izquierda
    dims_text = f"{bbox_width}x{bbox_height}px"
    cv2.putText(img, dims_text, (x1 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    
    # Dibujar landmarks mejorados
    cv2.circle(img, tuple(map(int, left_eye)), 8, landmark_color, -1)     # Ojo izq relleno
    cv2.circle(img, tuple(map(int, left_eye)), 8, (255, 200, 0), 2)       # Borde dorado
    
    cv2.circle(img, tuple(map(int, right_eye)), 8, landmark_color, -1)    # Ojo der relleno
    cv2.circle(img, tuple(map(int, right_eye)), 8, (255, 200, 0), 2)      # Borde dorado
    
    cv2.circle(img, tuple(map(int, nose)), 9, (100, 220, 255), -1)        # Nariz naranja claro MÁS VISIBLE
    cv2.circle(img, tuple(map(int, nose)), 9, (255, 200, 0), 2)           # Borde dorado más grueso
    
    cv2.circle(img, tuple(map(int, left_mouth)), 8, (100, 255, 255), -1)  # Boca amarilla CLARA MÁS VISIBLE
    cv2.circle(img, tuple(map(int, left_mouth)), 8, (255, 200, 0), 2)
    
    cv2.circle(img, tuple(map(int, right_mouth)), 8, (100, 255, 255), -1)
    cv2.circle(img, tuple(map(int, right_mouth)), 8, (255, 200, 0), 2)
    
    # Línea entre ojos mejorada
    cv2.line(img, tuple(map(int, left_eye)), tuple(map(int, right_eye)), landmark_color, 3)
    cv2.line(img, tuple(map(int, left_eye)), tuple(map(int, right_eye)), (255, 200, 0), 1)
    
    # Texto con IPD - Fondo elegante con degradado visual
    mid_x = int((left_eye[0] + right_eye[0]) / 2)
    mid_y = int((left_eye[1] + right_eye[1]) / 2) - 25
    text = f"IPD: {ipd:.1f}px"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    
    # Fondo elegante semi-transparente
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x_bg = mid_x - text_size[0] // 2 - 10
    y_bg = mid_y - text_size[1] - 10
    
    # Crear overlay para transparencia
    overlay = img.copy()
    cv2.rectangle(overlay, (x_bg, y_bg), (x_bg + text_size[0] + 20, y_bg + text_size[1] + 16), (50, 50, 100), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Borde elegante dorado
    cv2.rectangle(img, (x_bg, y_bg), (x_bg + text_size[0] + 20, y_bg + text_size[1] + 16), (255, 200, 0), 3)
    
    # Texto con sombra
    cv2.putText(img, text, (mid_x - text_size[0] // 2, mid_y), font, font_scale, (255, 255, 255), thickness)

# Guardar imagen
output = "visualizacion_landmarks.jpg"
cv2.imwrite(output, img)
print(f"✅ Imagen guardada: {output}")
