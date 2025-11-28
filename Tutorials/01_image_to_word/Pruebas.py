import os
import tempfile
from gtts import gTTS
import IPython.display as ipd
import pygame
from pygame import mixer
import time


class TextToSpeech:
    def __init__(self, language='es'):
        self.language = language

        # Inicializar el mezclador de audio de pygame
        try:
            mixer.init()
        except Exception as e:
            print(f"Nota: Mixer ya inicializado o no disponible: {e}")

    def text_to_speech(self, text, filename=None, slow=False):
        """Convierte texto a un archivo de audio MP3 usando Google TTS"""
        if not text or text.strip() == "":
            print("Texto vacío, no se puede generar audio")
            return None

        try:
            # Crear objeto gTTS
            tts = gTTS(text=text, lang=self.language, slow=slow)

            # Crear archivo temporal si no se especifica nombre
            if filename is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                filename = temp_file.name
                temp_file.close()

            # Guardar archivo de audio
            tts.save(filename)
            print(f"Audio guardado en: {filename}")
            return filename

        except Exception as e:
            print(f"Error al generar audio: {e}")
            return None

    def play_audio(self, audio_file):
        """Reproduce el archivo de audio generado"""
        if not audio_file or not os.path.exists(audio_file):
            print("Archivo de audio no encontrado.")
            return

        try:
            # Opción 1: Intentar usar IPython (ideal para Jupyter/Colab)
            try:
                display(ipd.Audio(audio_file, autoplay=True))
                return
            except NameError:
                pass  # No estamos en Jupyter/Colab

            # Opción 2: Usar Pygame (ideal para scripts locales)
            print("Reproduciendo audio con Pygame...")
            mixer.music.load(audio_file)
            mixer.music.play()

            # Esperar a que termine la reproducción
            while mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"Error al reproducir audio: {e}")


# --- EJEMPLO DE USO ---

if __name__ == "__main__":
    # 1. Configurar el motor
    tts_engine = TextToSpeech(language='es')  # 'es' para español

    # 2. Texto de prueba
    mi_texto = "Nunca ES FÁCIL IRSE Tenemos miedo a que se acaben cosas porque tene- más miedo a que no nos vuelvan a pasar: mos aún el no encontrar a alguien más que nos Tememos por lo menos igual que la última persona haga el día. Tenemos miedo de encontrarla y que nos alegaba voldea a sucios volver a pasar por %o mismo: s & salga mal, y de Tenemos miedo de que pase y de que no pase, 05 nadie lo supere. Y no nos damos da yde no encontrar a que cuenta de que solo nos vamos a arreglar cuando dejemos Jo tiene que ser 'un venga lo que nos de pensar las cosas, porque ello. Y venga\" lo vamos asumir y vamos a disfrutar de que Ilegue alguien que sea un desastre; que tenga mil defectos; que lo sepamos, y que no nos importe, que cada momento, con las que lleguen; con 19 las mucho poder que sentir superarlo idart hay ' porque disfrutar personas"

    # 3. Generar el archivo de audio
    archivo_generado = tts_engine.text_to_speech(mi_texto)

    # 4. Reproducir
    if archivo_generado:
        tts_engine.play_audio(archivo_generado)