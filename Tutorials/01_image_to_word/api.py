import cv2
import numpy as np
import os
import typing
import random
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import easyocr
import gtts
import base64
import io

# --- Construir rutas absolutas a los modelos ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Cargar modelos y configuraciones ---

# Modelo de detección de texto EAST
EAST_MODEL_PATH = os.path.join(SCRIPT_DIR, "frozen_east_text_detection.pb")
net = cv2.dnn.readNet(EAST_MODEL_PATH)

# Parámetros del modelo de detección
conf_threshold = 0.3
width, height = (1280, 1280)

# Capas de salida del modelo EAST
layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# Modelo de reconocimiento de palabras (ONNX)
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

# Cargar el vocabulario y el modelo ONNX
MODEL_VERSION = "202510191644"
MODEL_DIRECTORY = os.path.join(SCRIPT_DIR, "Models", "1_image_to_word", MODEL_VERSION)

configs = BaseModelConfigs.load(os.path.join(MODEL_DIRECTORY, "configs.yaml"))
recognition_model = ImageToWordModel(
    model_path=os.path.join(MODEL_DIRECTORY, "model.onnx"),
    char_list=configs.vocab
)

# Inicializar el lector de EasyOCR
print("Cargando el modelo de EasyOCR...")
easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
print("Modelos cargados. API lista.")


# --- Funciones de ayuda ---

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

def group_boxes_into_lines(boxes, tolerance_factor=0.4):
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        last_box_in_line = current_line[-1]
        last_box_center_y = (last_box_in_line[1] + last_box_in_line[3]) / 2
        current_box_center_y = (box[1] + box[3]) / 2
        last_box_height = last_box_in_line[3] - last_box_in_line[1]
        
        if abs(current_box_center_y - last_box_center_y) < last_box_height * tolerance_factor:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            
    lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines

# --- Aplicación FastAPI ---

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Ruta original: Usa el modelo mltu, genera audio y devuelve ambos.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    (H, W) = orig.shape[:2]
    debug_image = orig.copy()
    rW = W / float(width)
    rH = H / float(height)
    image = cv2.resize(orig, (width, height))
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    (rects, confidences) = decode_predictions(scores, geometry, conf_threshold)
    box_indices = cv2.dnn.NMSBoxes(
        bboxes=[[x, y, ex - x, ey - y] for (x, y, ex, ey) in rects],
        scores=confidences,
        score_threshold=conf_threshold,
        nms_threshold=0.4
    )
    if box_indices is None or len(box_indices) == 0:
        return JSONResponse(content={"text": "", "audio_base64": ""})
        
    final_boxes = [rects[i] for i in box_indices.flatten()]
    lines = group_boxes_into_lines(final_boxes)
    line_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(lines))]
    all_lines_text = []
    padding = 4
    for i, line_boxes in enumerate(lines):
        line_words = []
        color = line_colors[i]
        for (startX, startY, endX, endY) in line_boxes:
            startX_orig = int(startX * rW) - padding
            startY_orig = int(startY * rH) - padding
            endX_orig = int(endX * rW) + padding
            endY_orig = int(endY * rH) + padding
            startX_orig, startY_orig = max(0, startX_orig), max(0, startY_orig)
            endX_orig, endY_orig = min(W, endX_orig), min(H, endY_orig)
            cv2.rectangle(debug_image, (startX_orig, startY_orig), (endX_orig, endY_orig), color, 2)
            cropped = orig[startY_orig:endY_orig, startX_orig:endX_orig]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                try:
                    prediction_text = recognition_model.predict(cropped)
                    line_words.append(prediction_text)
                except Exception as e:
                    print(f"Error al predecir palabra: {e}")
                    continue
        all_lines_text.append(" ".join(line_words))
        
    debug_image_path = os.path.join(SCRIPT_DIR, "debug_lines.png")
    cv2.imwrite(debug_image_path, debug_image)
    print(f"Imagen de depuración guardada en: {debug_image_path}")
    
    full_text = "\n".join(all_lines_text)
    
    # --- Lógica de generación de audio ---
    audio_base64 = ""
    if full_text.strip():
        try:
            tts = gtts.gTTS(full_text, lang='es')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_bytes = mp3_fp.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error al generar el audio con gTTS: {e}")

    return JSONResponse(content={"text": full_text, "audio_base64": audio_base64})

@app.post("/predict2/")
async def predict2_image(file: UploadFile = File(...)):
    """
    Usa EasyOCR para extraer texto y gTTS para generar un audio a partir del texto.
    Devuelve el texto y el audio codificado en Base64.
    """
    contents = await file.read()
    results = easyocr_reader.readtext(contents)
    
    extracted_texts = [res[1] for res in results]
    full_text = "\n".join(extracted_texts)
    
    audio_base64 = ""
    if full_text.strip():
        try:
            tts = gtts.gTTS(full_text, lang='es')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_bytes = mp3_fp.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error al generar el audio con gTTS: {e}")
    
    return JSONResponse(content={"text": full_text, "audio_base64": audio_base64})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
