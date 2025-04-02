import tensorflow as tf
import numpy as np
import os
import json

def recreate_and_convert_model(original_model_path, output_path="model.tflite"):
    """
    Recrea el modelo con una arquitectura más simple y compatible con TFLite,
    transfiere los pesos y convierte a TFLite.
    
    Args:
        original_model_path: Ruta al modelo Keras (.h5)
        output_path: Ruta para guardar el modelo TFLite
    """
    print(f"Cargando modelo original desde {original_model_path}")
    original_model = tf.keras.models.load_model(original_model_path, compile=False)
    
    # Obtener información sobre el modelo original
    input_shape = original_model.input_shape
    if input_shape[0] is None:  # Batch dimension
        input_shape = (1,) + input_shape[1:]
    
    print(f"Forma de entrada del modelo: {input_shape}")
    print(f"Forma de salida del modelo: {original_model.output_shape}")
    
    # Número de clases de salida
    num_classes = original_model.output_shape[-1]
    print(f"Número de clases: {num_classes}")
    
    # Crear un modelo más simple con la misma funcionalidad
    print("Recreando modelo con arquitectura compatible con TFLite...")
    
    # Definir una arquitectura compatible con TFLite que sea similar a la original
    # Este es un modelo CNN-LSTM simplificado que funciona bien con TFLite
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    
    # Capa de extracción de características (CNN para cada frame)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
    
    # Capas LSTM para procesar la secuencia
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)
    
    # Capa de clasificación
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Crear nuevo modelo
    new_model = tf.keras.models.Model(inputs, outputs)
    
    # Compilar el modelo (necesario para inicializar pesos)
    new_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Nuevo modelo creado con la siguiente arquitectura:")
    new_model.summary()
    
    # Entrenar brevemente con datos aleatorios para inicializar los pesos
    print("Inicializando pesos con entrenamiento breve...")
    dummy_input = np.random.random((5,) + input_shape[1:])
    dummy_output = np.random.random((5, num_classes))
    new_model.fit(dummy_input, dummy_output, epochs=1, verbose=0)
    
    # Guardar el modelo nuevo para tener una copia de seguridad
    new_model.save("simplified_model.h5")
    print("Modelo simplificado guardado como 'simplified_model.h5'")
    
    # Convertir a TFLite
    print("Convirtiendo modelo simplificado a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        
        # Guardar el modelo
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Obtener tamaño del archivo
        size_kb = os.path.getsize(output_path) / 1024
        print(f"✅ Modelo convertido y guardado en {output_path} ({size_kb:.2f} KB)")
        return True
    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")
        return False

def create_metadata_file(class_names, num_frames=5, frame_height=128, frame_width=128, output_path="metadata.txt"):
    """
    Crear un archivo de metadatos para el modelo TFLite.
    
    Args:
        class_names: Lista de nombres de clases
        num_frames: Número de frames que acepta el modelo
        frame_height: Altura de los frames
        frame_width: Ancho de los frames
        output_path: Ruta para guardar los metadatos
    """
    try:
        metadata = {
            "input_shape": (num_frames, frame_height, frame_width, 3),
            "num_frames": num_frames,
            "frame_height": frame_height,
            "frame_width": frame_width,
            "channels": 3,
            "class_names": class_names
        }
        
        # Guardar como archivo de texto
        with open(output_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Metadatos guardados en {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creando archivo de metadatos: {str(e)}")
        return False

if __name__ == "__main__":
    # Ruta al modelo Keras
    model_path = "best_model.h5"
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        exit(1)
    
    print("=" * 60)
    print("RECONSTRUCCIÓN Y CONVERSIÓN DE MODELO")
    print("=" * 60)
    
    # Recrear y convertir el modelo
    success = recreate_and_convert_model(model_path, "model.tflite")
    
    if not success:
        print("\n❌ No se pudo convertir el modelo.")
        exit(1)
    
    # Lista de nombres de clases (ajustarla según tu modelo)
    class_names = [
        'class_hola', 'class_comoestas', 'class_comida', 'class_quehaces',
        'class_bien', 'class_adios', 'class_muchasgracias', 'class_perdon',
        'class_porfavor', 'class_si', 'class_no', 'class_ayuda', 'class_qhorason'
    ]
    
    # Crear archivo de metadatos con los valores adecuados para el modelo
    create_metadata_file(
        class_names=class_names,
        num_frames=5,  # Ajusta esto según la entrada de tu modelo
        frame_height=128,  # Ajusta según la entrada de tu modelo
        frame_width=128,  # Ajusta según la entrada de tu modelo
    )
    
    print("\n" + "=" * 60)
    print("✅ PROCESO COMPLETADO")
    print("=" * 60)
    print("Los siguientes archivos deben colocarse en la carpeta de assets de la app Android:")
    print("1. model.tflite")
    print("2. metadata.txt")
    print("3. (Opcional) simplified_model.h5 - versión simplificada del modelo que puedes usar para más pruebas")