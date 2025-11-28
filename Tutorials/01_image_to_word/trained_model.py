import cv2
import random
import numpy as np
import typing # <--- AÑADIDO PARA SOLUCIONAR EL ERROR
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
# Importar la clase base Transformer para crear las nuestras
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, Transformer
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from configs import ModelConfigs

# --- Definición de nuestras propias clases de Data Augmentation ---

class RandomRotate(Transformer):
    def __init__(self, angle: int = 5):
        super().__init__()
        self.angle = angle

    def __call__(self, image: CVImage, label: typing.Any):
        angle = random.randint(-self.angle, self.angle)
        (h, w) = image.numpy().shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image.numpy(), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image.update(rotated)
        return image, label

class RandomBrightness(Transformer):
    def __init__(self, max_delta: float = 0.3):
        super().__init__()
        self.max_delta = max_delta

    def __call__(self, image: CVImage, label: typing.Any):
        delta = random.uniform(-self.max_delta, self.max_delta)
        hsv = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int(delta * 255))
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        image.update(bright_image)
        return image, label

class RandomErodeDilate(Transformer):
    def __init__(self):
        super().__init__()

    def __call__(self, image: CVImage, label: typing.Any):
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            # Erosión (adelgaza las letras)
            eroded = cv2.erode(image.numpy(), kernel, iterations=1)
            image.update(eroded)
        else:
            # Dilatación (engrosa las letras)
            dilated = cv2.dilate(image.numpy(), kernel, iterations=1)
            image.update(dilated)
        return image, label

# -------------------------------------------------------------------

configs = ModelConfigs()

# Cargar el modelo previamente entrenado
model = load_model("Models/1_image_to_word/202510191644/model.h5", compile=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=0)],
    run_eagerly=False
)
model.summary(line_length=110)

data_path = r"C:/Users/camil/Desktop/90kDICT32px"
val_annotation_path = data_path + "/annotation_val.txt"
train_annotation_path = data_path + "/annotation_train.txt"

# Read metadata file and parse it
def read_annotation_file(annotation_path):
    dataset, vocab, max_len = [], set(), 0
    with open(annotation_path, "r") as f:
        for line in tqdm(f.readlines()):
            # Limitar el dataset para pruebas más rápidas si es necesario
            if len(dataset) <= 300000:
                line = line.split()
                image_path = data_path + line[0][1:]
                label = line[0].split("_")[1]
                dataset.append([image_path, label])
                vocab.update(list(label))
                max_len = max(max_len, len(label))
    return dataset, sorted(vocab), max_len

train_dataset, train_vocab, max_train_len = read_annotation_file(train_annotation_path)
val_dataset, val_vocab, max_val_len = read_annotation_file(val_annotation_path)

configs.vocab = "".join(train_vocab)
configs.max_text_length = max(max_train_len, max_val_len)
configs.save()

train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # Aquí usamos nuestras clases personalizadas de Data Augmentation
        RandomBrightness(),
        RandomRotate(angle=3),
        RandomErodeDilate(),
        # El resto de transformadores
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

# El val_data_provider NO debe tener data augmentation
val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

# Callbacks
# Usar la versión más reciente del modelo para guardar los nuevos resultados


earlystopper = EarlyStopping(monitor="val_CER", patience=4, verbose=1)
checkpoint = ModelCheckpoint("Models/1_image_to_word/202510191644/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=2, verbose=1, mode="auto")
trainLogger = TrainLogger("Models/1_image_to_word/202510191644")
tb_callback = TensorBoard("Models/1_image_to_word/202510191644/logs", update_freq=1)
model2onnx = Model2onnx("Models/1_image_to_word/202510191644/model.h5")

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=20, # Puedes ajustar las épocas
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)
