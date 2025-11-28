from tensorflow.keras.models import load_model
import os
from tqdm import tqdm
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric


from model import train_model
from configs import ModelConfigs

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
            if len(dataset) <= 100000:
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
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

# os.makedirs("Models/1_image_to_word/202510191644/model.h5", exist_ok=True)

earlystopper = EarlyStopping(monitor="val_CER", patience=10, verbose=1)
checkpoint = ModelCheckpoint("Models/1_image_to_word/202510191644/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="auto")
trainLogger = TrainLogger("Models/1_image_to_word/202510191644")
tb_callback = TensorBoard("Models/1_image_to_word/202510191644/logs", update_freq=1)
model2onnx = Model2onnx("Models/1_image_to_word/202510191644/model.h5")

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=15,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)