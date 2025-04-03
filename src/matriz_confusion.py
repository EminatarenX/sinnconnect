import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tqdm import tqdm

# Configurar TensorFlow
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Usando aceleración GPU")
    else:
        print("No se encontró GPU, usando CPU")
except Exception as e:
    print(f"Error configurando GPU: {e}")

def cargar_modelo(ruta_modelo='best_model.h5'):
    """
    Carga el modelo previamente entrenado.
    
    Args:
        ruta_modelo: Ruta al archivo del modelo entrenado (.h5)
        
    Returns:
        modelo: Modelo cargado
    """
    print(f"Cargando modelo desde {ruta_modelo}...")
    modelo = tf.keras.models.load_model(ruta_modelo)
    print("Modelo cargado exitosamente")
    return modelo

def cargar_metadatos(ruta_archivo="model_metadata.txt"):
    """
    Carga metadatos del modelo, incluyendo los nombres de las clases.
    
    Args:
        ruta_archivo: Ruta al archivo de metadatos
        
    Returns:
        nombres_clases: Lista de nombres de clases
    """
    nombres_clases = []
    
    try:
        if os.path.exists(ruta_archivo):
            with open(ruta_archivo, 'r') as f:
                lineas = f.readlines()
                
            for linea in lineas:
                if "class_names" in linea:
                    # Extraer lista de nombres de clase
                    valor = linea.split(":", 1)[1].strip()
                    valor = valor.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                    nombres_clases = [c.strip() for c in valor.split(",")]
                    break
            
            print(f"Clases cargadas: {nombres_clases}")
            return nombres_clases
        else:
            print(f"Archivo {ruta_archivo} no encontrado.")
            # Define clases por defecto si no se puede cargar el archivo
            return ['class_hola', 'class_comoestas', 'class_comida', 'class_quehaces', 'class_bien', 
                    'class_adios', 'class_muchasgracias', 'class_perdon', 'class_porfavor', 
                    'class_si', 'class_no', 'class_ayuda', 'class_qhorason']
    except Exception as e:
        print(f"Error cargando metadatos: {e}")
        return []

def cargar_datos_validacion(dataset_dir, img_size=(128, 128), n_frames=5):
    """
    Carga un conjunto de datos para validación y pruebas.
    
    Args:
        dataset_dir: Ruta al directorio del dataset
        img_size: Tamaño de imagen (ancho, alto)
        n_frames: Número de frames por secuencia
        
    Returns:
        X: Datos de secuencias
        y: Etiquetas
        nombres_clases: Lista de nombres de clases
    """
    print(f"Cargando datos de validación desde {dataset_dir}...")
    
    # Verificar si el directorio existe
    if not os.path.exists(dataset_dir):
        print(f"ERROR: El directorio {dataset_dir} no existe")
        return None, None, []
    
    # Obtener nombres de clases
    nombres_clases = os.listdir(dataset_dir)
    nombres_clases = [d for d in nombres_clases if os.path.isdir(os.path.join(dataset_dir, d))]
    nombres_clases.sort()  # Ordenar para indexación consistente
    
    X = []
    y = []
    
    for idx_clase, nombre_clase in enumerate(tqdm(nombres_clases, desc="Cargando clases")):
        ruta_clase = os.path.join(dataset_dir, nombre_clase)
        
        # Obtener todos los sets (videos)
        sets = os.listdir(ruta_clase)
        sets = [s for s in sets if os.path.isdir(os.path.join(ruta_clase, s))]
        
        for nombre_set in sets:
            ruta_set = os.path.join(ruta_clase, nombre_set)
            
            # Obtener todas las secuencias en este set
            secuencias = os.listdir(ruta_set)
            secuencias = [s for s in secuencias if os.path.isdir(os.path.join(ruta_set, s))]
            
            for nombre_secuencia in secuencias:
                ruta_secuencia = os.path.join(ruta_set, nombre_secuencia)
                
                # Cargar todos los frames en esta secuencia
                frames = []
                archivos_frame = [f for f in os.listdir(ruta_secuencia) if f.endswith('.jpg')]
                archivos_frame.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ordenar por número de frame
                
                for archivo_frame in archivos_frame:
                    ruta_frame = os.path.join(ruta_secuencia, archivo_frame)
                    img = cv2.imread(ruta_frame)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
                    img = cv2.resize(img, img_size)
                    frames.append(img)
                
                # Asegurar que tenemos exactamente n_frames
                if len(frames) == n_frames:
                    X.append(frames)
                    y.append(idx_clase)
    
    # Convertir listas a arrays numpy
    X = np.array(X, dtype=np.float32) / 255.0  # Normalizar valores de píxeles
    y = tf.keras.utils.to_categorical(y)  # Codificar etiquetas en one-hot
    
    print(f"Datos cargados: {len(X)} muestras, {len(nombres_clases)} clases")
    return X, y, nombres_clases

def generar_matriz_confusion(modelo, X_test, y_test, nombres_clases):
    """
    Genera y grafica la matriz de confusión.
    
    Args:
        modelo: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba (codificadas en one-hot)
        nombres_clases: Lista de nombres de clases
    """
    print("Generando predicciones para la matriz de confusión...")
    y_pred = modelo.predict(X_test, verbose=1)
    
    # Convertir de codificación one-hot a clases
    y_test_clases = np.argmax(y_test, axis=1)
    y_pred_clases = np.argmax(y_pred, axis=1)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test_clases, y_pred_clases)
    
    # Crear nombres más legibles
    etiquetas_clase = [nombre.replace('class_', '') for nombre in nombres_clases]
    
    # Normalizar matriz para mejor visualización
    cm_normalizada = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Configurar figura
    plt.figure(figsize=(14, 12))
    
    # Matriz sin normalizar
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas_clase, yticklabels=etiquetas_clase)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    # Matriz normalizada
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalizada, annot=True, fmt='.2f', cmap='Blues', xticklabels=etiquetas_clase, yticklabels=etiquetas_clase)
    plt.title('Matriz de Confusión (Normalizada)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=300)
    plt.show()
    
    # Imprimir reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(y_test_clases, y_pred_clases, target_names=etiquetas_clase))
    
    # Calcular precisión general
    precision = np.mean(y_test_clases == y_pred_clases)
    print(f"\nPrecisión general: {precision:.4f}")

if __name__ == "__main__":
    # Ruta al directorio del dataset
    dataset_dir = "../data/dataset"
    
    # Ruta al modelo entrenado
    ruta_modelo = "best_model.h5"
    
    # Verificar si existe el modelo
    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encontró el modelo en {ruta_modelo}")
        # Buscar archivos .h5 en el directorio actual
        archivos_h5 = [f for f in os.listdir('.') if f.endswith('.h5')]
        if archivos_h5:
            ruta_modelo = archivos_h5[0]
            print(f"Se usará el modelo: {ruta_modelo}")
        else:
            print("No se encontraron modelos .h5 en el directorio actual.")
            print("Por favor, especifica la ruta correcta del modelo.")
            exit(1)
    
    # Cargar modelo previamente entrenado
    modelo = cargar_modelo(ruta_modelo)
    
    # Cargar metadatos (nombres de clases)
    nombres_clases = cargar_metadatos()
    
    # Si no se pudieron cargar nombres de clases, intentar inferirlos del dataset
    if not nombres_clases:
        print("No se pudieron cargar nombres de clases desde metadatos.")
        print("Intentando inferir nombres desde la estructura del dataset...")
        _, _, nombres_clases = cargar_datos_validacion(dataset_dir, max_samples=1)
    
    # Cargar datos para evaluación
    X_test, y_test, _ = cargar_datos_validacion(dataset_dir)
    
    if X_test is not None and y_test is not None:
        # Generar y mostrar matriz de confusión
        generar_matriz_confusion(modelo, X_test, y_test, nombres_clases)
    else:
        print("No se pudieron cargar los datos para evaluación.")