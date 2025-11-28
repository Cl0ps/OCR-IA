import cv2
import numpy as np
import imutils
import os

# Ruta al modelo preentrenado EAST
EAST_MODEL = "frozen_east_text_detection.pb"

# Ruta a la imagen que quieres procesar
IMAGE_PATH = "imagenes/imagen3.jpeg"

# Directorio donde guardar los recortes
OUTPUT_DIR = "recortes_palabras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros del modelo
conf_threshold = 0.5
width, height = (640, 640)  # Puedes probar también (640, 640) para más precisión

# Cargar imagen
image = cv2.imread(IMAGE_PATH)
orig = image.copy()
(H, W) = image.shape[:2]

# Calcular proporción de cambio de tamaño
rW = W / float(width)
rH = H / float(height)

# Redimensionar imagen para el modelo EAST
image = cv2.resize(image, (width, height))
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

# Cargar red EAST
net = cv2.dnn.readNet(EAST_MODEL)

# Capas de salida
layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# Ejecutar detección
net.setInput(blob)
(scores, geometry) = net.forward(layer_names)

# --- Función para decodificar resultados del EAST ---
def decode_predictions(scores, geometry, confThreshold):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < confThreshold:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return (rects, confidences)

# Decodificar predicciones
(rects, confidences) = decode_predictions(scores, geometry, conf_threshold)

# Aplicar supresión no máxima para eliminar solapamientos
boxes = cv2.dnn.NMSBoxes(
    bboxes=[[x, y, ex - x, ey - y] for (x, y, ex, ey) in rects],
    scores=confidences,
    score_threshold=conf_threshold,
    nms_threshold=0.4
)

# Ajustar cajas a tamaño original y recortar
count = 0
for i in boxes:
    (startX, startY, endX, endY) = rects[i]
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Asegurar que no salgan de los límites
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(W, endX), min(H, endY)

    # Recortar palabra
    cropped = orig[startY:endY, startX:endX]
    output_path = os.path.join(OUTPUT_DIR, f"word_{count:03d}.png")
    cv2.imwrite(output_path, cropped)
    count += 1

    # Dibujar la caja en la imagen original (opcional)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

print(f"✅ {count} palabras detectadas y guardadas en '{OUTPUT_DIR}'")

# Mostrar resultado
cv2.imshow("Detección de texto", imutils.resize(orig, width=800))
cv2.waitKey(0)
cv2.destroyAllWindows()
