#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import json
import insightface
from tqdm import tqdm
from pathlib import Path

IMAGE_DIR = "/home/capistran/Documents/download_gs_path/streamax_asset_126414_2/"
OUTPUT_FILE = "resultados.json"

# Inicializar InsightFace (usa RetinaFace internamente)
app = insightface.app.FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0)

images = sorted(glob.glob(f"{IMAGE_DIR}*.jpeg"))
all_distances = []
ipd_per_image = []  # Registrar (ipd, filename)

print(f"📁 Directorio: {IMAGE_DIR}")
print(f"🖼️  Imágenes encontradas: {len(images)}")
print(f"🔍 Procesando con RetinaFace (vía InsightFace)...\n")

for img_path in tqdm(images, desc="Analizando"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # RetinaFace detector integrado en InsightFace
        faces = app.get(img)
        
        for face in faces:
            # 5 landmarks básicos: [left_eye, right_eye, nose, left_mouth, right_mouth]
            kps = face.kps
            left_eye = kps[0]
            right_eye = kps[1]
            
            ipd = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
            all_distances.append(ipd)
            ipd_per_image.append((ipd, Path(img_path).name))
    except:
        pass

# Resultados
print("\n" + "="*60)
print("📊 RESULTADOS")
print("="*60)
print(f"Total mediciones: {len(all_distances)}")

if all_distances:
    dist_array = np.array(all_distances)
    media = np.mean(dist_array)
    std = np.std(dist_array)
    
    min_ipd = np.min(dist_array)
    max_ipd = np.max(dist_array)
    
    print(f"\n📏 Distancia Interpupilar:")
    print(f"  Media:                 {media:.4f} píxeles")
    print(f"  Desviación Estándar:   {std:.4f} píxeles")
    print(f"  Mínimo:                {min_ipd:.4f} píxeles")
    print(f"  Máximo:                {max_ipd:.4f} píxeles")
    print(f"  Mediana:               {np.median(dist_array):.4f} píxeles")
    
    # Encontrar imágenes con mínimo y máximo IPD
    min_idx = np.argmin(dist_array)
    max_idx = np.argmax(dist_array)
    
    min_img = ipd_per_image[min_idx][1]
    max_img = ipd_per_image[max_idx][1]
    
    print(f"\n🎯 IPD EXTREMOS:")
    print(f"  Mínimo: {min_ipd:.4f} px → {min_img}")
    print(f"  Máximo: {max_ipd:.4f} px → {max_img}")
    
    # Guardar
    results = {
        'total_mediciones': len(all_distances),
        'media_pixeles': float(media),
        'desviacion_estandar_pixeles': float(std),
        'minimo_pixeles': float(min_ipd),
        'maximo_pixeles': float(max_ipd),
        'mediana_pixeles': float(np.median(dist_array)),
        'imagen_minimo_ipd': min_img,
        'imagen_maximo_ipd': max_img
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Guardado en: {OUTPUT_FILE}")
else:
    print("❌ Sin datos")

print("="*60)
