import easyocr
import cv2
import os

# --- 1. Definir la ruta a la imagen ---
# Asegúrate de que la imagen exista en esta ruta.
IMAGE_PATH = 'imagenes/imagen3.jpeg'

# --- 2. Inicializar el lector de EasyOCR ---
# La primera vez que se ejecuta, descargará los modelos para los idiomas especificados.
# Usamos ['es', 'en'] para que pueda reconocer tanto español como inglés.
print("Cargando el modelo de EasyOCR... (puede tardar un momento la primera vez)")
reader = easyocr.Reader(['es', 'en'], gpu=True) # gpu=True para usar la GPU si está configurada, si no, cambiar a False.

# --- 3. Cargar y procesar la imagen ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: No se encontró la imagen en la ruta '{IMAGE_PATH}'")
else:
    # Leer la imagen con OpenCV
    image = cv2.imread(IMAGE_PATH)
    
    # Realizar el reconocimiento de texto
    print("Procesando la imagen para extraer texto...")
    results = reader.readtext(image)
    
    # --- 4. Imprimir y visualizar los resultados ---
    print("\n--- Texto extraído ---")
    
    # Crear una copia de la imagen para dibujar los resultados
    output_image = image.copy()
    
    for (bbox, text, prob) in results:
        # Imprimir cada texto detectado y su confianza
        print(f'Texto: "{text}", Confianza: {prob:.4f}')
        
        # bbox es una lista de 4 puntos [top-left, top-right, bottom-right, bottom-left]
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # Dibujar el rectángulo delimitador en la imagen
        cv2.rectangle(output_image, tl, br, (0, 255, 0), 2)
        
        # Poner el texto reconocido cerca de la caja
        cv2.putText(output_image, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("----------------------\n")
    
    # Guardar la imagen con los resultados
    output_path = 'ocr_result.png'
    cv2.imwrite(output_path, output_image)
    print(f"Resultados visuales guardados en: '{output_path}'")
    
    # Mostrar la imagen con los resultados
    cv2.imshow("Resultado de OCR", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
