from tensorflow.keras import layers, models

def detection_model(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape)

    # Backbone (extractor de características)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    # Bloques adicionales
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)

    # Salida: mapa de probabilidad de texto (1 canal)
    output = layers.Conv2D(1, (1,1), activation='sigmoid', name='text_score')(x)

    model = models.Model(inputs=inputs, outputs=output, name="TextDetectionModel")
    return model
