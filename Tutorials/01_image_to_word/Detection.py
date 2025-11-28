import cv2
import numpy as np
import imutils
import os

from skimage.util import img_as_bool

# Ruta al modelo preentrenado EAST
EAST_MODEL = "frozen_east_text_detection.pb"

# Ruta a la imagen que quieres procesar
IMAGE_PATH = "imagenes/imagen3.jpeg"

# Directorio donde guardar los recortes
OUTPUT_DIR = f"recortes_palabras/{IMAGE_PATH.split('.')[0]}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros del modelo
conf_threshold = 0.3
width, height = (1280, 1280)  # Puedes probar también (640, 640) para más precisión

# Cargar imagen
image = cv2.imread(IMAGE_PATH)
orig = image.copy()
img_mostrar = image.copy()
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

def segment_words(line_image):
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilatar para unir letras dentro de cada palabra
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Buscar contornos de palabras
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        word_boxes.append((x, y, x + w, y + h))

    # Ordenar de izquierda a derecha
    word_boxes = sorted(word_boxes, key=lambda b: b[0])
    return word_boxes
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
allimgs = []
padding = 1
for i in boxes:
    (startX, startY, endX, endY) = rects[i]
    startX = int(startX * rW)  - padding
    startY = int(startY * rH) - padding - 1
    endX = int(endX * rW) + padding
    endY = int(endY * rH) + padding + 2

    # Asegurar que no salgan de los límites
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(W, endX), min(H, endY)

    # Recortar palabra
    cropped = orig[startY:endY, startX:endX]
    output_path = os.path.join(OUTPUT_DIR, f"word_{count:03d}.png")
    allimgs.append(output_path)
    cv2.imwrite(output_path, cropped)
    count += 1

    # Dibujar la caja en la imagen original (opcional)
    cv2.rectangle(img_mostrar, (startX, startY), (endX, endY), (0, 255, 0), 2)


with open(f"recortes_palabras/{IMAGE_PATH.split('.')[0]}/imagenes.txt", "w", encoding="utf-8") as f:
    for image in allimgs:
        image = image.replace("\\","/")
        f.write(f"{image}\n")

print(f"✅ {count} palabras detectadas y guardadas en '{OUTPUT_DIR}'")

# Mostrar resultado
cv2.imshow("Detección de texto", imutils.resize(img_mostrar, width=800))
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################

import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    directory = "202510191644"

    configs = BaseModelConfigs.load(f"Models/1_image_to_word/{directory}/configs.yaml")

    model = ImageToWordModel(model_path=f"Models/1_image_to_word/{directory}/model.onnx", char_list=configs.vocab)

    df = pd.read_csv(f"Models/1_image_to_word/{directory}/val.csv").dropna().values.tolist()

    # accum_cer = []
    for image_path in allimgs:
        image = cv2.imread(image_path.replace("\\", "/"))

        try:
            prediction_text = model.predict(image)

            # cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Prediction: {prediction_text}")

            # resize image by 3 times for visualization
            image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
            cv2.imshow(prediction_text, image)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()
        except:
            continue

        # accum_cer.append(cer)

    # print(f"Average CER: {np.average(accum_cer)}")
