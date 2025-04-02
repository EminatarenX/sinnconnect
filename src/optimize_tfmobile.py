import tensorflow as tf
import numpy as np
import os
import shutil

def export_saved_model_for_android(model_path, output_dir="saved_model"):
    """
    Exporta un modelo Keras para ser usado con TensorFlow Mobile en Android.
    
    Args:
        model_path: Ruta al modelo Keras (.h5)
        output_dir: Directorio para guardar el modelo exportado
    """
    print(f"Cargando modelo desde {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Obtener información sobre el modelo
    input_shape = model.input_shape
    if input_shape[0] is None:  # Batch dimension
        input_shape = (1,) + input_shape[1:]
    
    print(f"Forma de entrada del modelo: {input_shape}")
    print(f"Forma de salida del modelo: {model.output_shape}")
    
    # Crear un directorio para el modelo exportado
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Definir una función que sirva como punto de entrada
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
    def serving_function(input_tensor):
        return model(input_tensor)
    
    # Crear signatures para la exportación
    signatures = {
        "serving_default": serving_function
    }
    
    # Exportar el modelo
    print(f"Exportando modelo a {output_dir}...")
    try:
        tf.saved_model.save(model, output_dir, signatures=signatures)
        print(f"✅ Modelo exportado correctamente a {output_dir}")
        
        # Comprimir modelo para facilitar su distribución
        shutil.make_archive(output_dir, 'zip', output_dir)
        print(f"✅ Modelo comprimido como {output_dir}.zip")
        
        return True
    except Exception as e:
        print(f"❌ Error exportando el modelo: {str(e)}")
        return False

def create_metadata_file(model_path, class_names, output_path="metadata.txt"):
    """
    Crear un archivo de metadatos para el modelo.
    
    Args:
        model_path: Ruta al modelo Keras (.h5)
        class_names: Lista de nombres de clases
        output_path: Ruta para guardar los metadatos
    """
    try:
        print(f"Cargando modelo para obtener metadatos: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Obtener información de forma de entrada
        input_shape = model.input_shape[1:]  # Omitir dimensión de batch
        print(f"Forma de entrada: {input_shape}")
        
        metadata = {
            "input_shape": input_shape,
            "num_frames": input_shape[0],
            "frame_height": input_shape[1],
            "frame_width": input_shape[2],
            "channels": input_shape[3],
            "class_names": class_names,
            "input_name": "serving_default_input_1",
            "output_name": "StatefulPartitionedCall"
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
    print("EXPORTACIÓN DE MODELO PARA ANDROID")
    print("=" * 60)
    
    # Exportar el modelo
    success = export_saved_model_for_android(model_path, "saved_model")
    
    if not success:
        print("\n❌ No se pudo exportar el modelo.")
        exit(1)
    
    # Lista de nombres de clases (ajustarla según tu modelo)
    class_names = [
        'class_hola', 'class_comoestas', 'class_comida', 'class_quehaces',
        'class_bien', 'class_adios', 'class_muchasgracias', 'class_perdon',
        'class_porfavor', 'class_si', 'class_no', 'class_ayuda', 'class_qhorason'
    ]
    
    # Crear archivo de metadatos
    create_metadata_file(model_path, class_names, "metadata.txt")
    
    print("\n" + "=" * 60)
    print("✅ PROCESO COMPLETADO")
    print("=" * 60)
    print("Para usar con Android:")
    print("1. Coloca saved_model.zip en la carpeta 'assets' de tu proyecto Android")
    print("2. Descomprime el modelo en tiempo de ejecución en la app")
    print("3. Usa TensorFlow Mobile para cargar el modelo descomprimido")