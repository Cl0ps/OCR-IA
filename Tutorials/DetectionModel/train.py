import os
from tqdm import tqdm
import tensorflow as tf

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import detection_model
from configs import ModelConfigs
import pandas as pd
import ast
import cv2

configs = ModelConfigs()

data_path = r"C:/Users/camil/Desktop/DetectionSet"
val_annotation_path = data_path + "/img.csv"
train_annotation_path = data_path + "/annot.csv"

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import ast
import os

# =============================
# CONFIGURACIÓN GENERAL
# =============================
ANNOT_PATH = "C:/Users/camil/Desktop/DetectionSet/annot.csv"
IMG_PATH = "C:/Users/camil/Desktop/DetectionSet/img.csv"
BATCH_SIZE = 512
EPOCHS = 50
IMG_SIZE = (512, 512)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# CARGA Y PREPARACIÓN DEL DATASET
# =============================
print("[INFO] Cargando CSVs...")
df_annot = pd.read_csv(ANNOT_PATH)
df_imgs = pd.read_csv(IMG_PATH)

# Unimos anotaciones con rutas de imágenes
df = df_annot.merge(df_imgs, left_on="image_id", right_on="id")


# Función para leer imágenes y etiquetas
def load_image_and_labels(row):
    image = tf.io.read_file(row["file_name"])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    bbox = tf.convert_to_tensor(ast.literal_eval(row["bbox"]), dtype=tf.float32)
    bbox = bbox / tf.constant([row["width"], row["height"], row["width"], row["height"]], dtype=tf.float32)

    label = tf.constant([1.0], dtype=tf.float32)  # clase: texto
    return image, (bbox, label)


# Convertimos DataFrame a tf.data.Dataset
def make_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.map(lambda x: load_image_and_labels(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Separar train/val
train_df = df[df["set"] == "train"]
val_df = df[df["set"] == "val"]

train_ds = make_dataset(train_df)
val_ds = make_dataset(val_df)


# =============================
# DEFINICIÓN DEL MODELO
# =============================
def simple_text_detector():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)

    bbox = layers.Dense(4, activation='sigmoid', name='bbox')(x)
    cls = layers.Dense(1, activation='sigmoid', name='cls')(x)

    return models.Model(inputs, [bbox, cls])


model = simple_text_detector()
model.summary()

# =============================
# COMPILACIÓN Y ENTRENAMIENTO
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={'bbox': 'mse', 'cls': 'binary_crossentropy'},
    metrics={'cls': 'accuracy'}
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "text_detector.keras"),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
]

print("[INFO] Iniciando entrenamiento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("[INFO] Entrenamiento finalizado. Modelo guardado en:", MODEL_DIR)
