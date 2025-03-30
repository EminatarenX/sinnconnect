import os
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt

def load_keras_model(model_path):
    """
    Load a Keras model.
    
    Args:
        model_path: Path to the Keras model (.h5)
        
    Returns:
        model: Loaded Keras model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_class_names(metadata_path):
    """
    Load class names from the model metadata file.
    
    Args:
        metadata_path: Path to the model metadata file
        
    Returns:
        class_names: List of class names
    """
    class_names = []
    try:
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("class_names:"):
                    # Extract class names from the line
                    classes_str = line.split(':')[1].strip()
                    # Clean up the format
                    classes_str = classes_str.replace('[', '').replace(']', '').replace("'", "")
                    # Split by commas and strip whitespace
                    class_names = [c.strip() for c in classes_str.split(',')]
                    break
        
        print(f"Loaded {len(class_names)} class names: {class_names}")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to class indices if we can't load names
        return [f"class_{i}" for i in range(13)]

def preprocess_video(video_path, num_frames=10, target_size=(224, 224)):
    """
    Preprocess a video for inference.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        target_size: Target frame size (width, height)
        
    Returns:
        sequence: Preprocessed sequence of frames
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {frame_count} frames")
    
    # We'll use the same frame selection logic as in dataset creation
    base_proportions = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    
    # Calculate frame indices
    frame_indices = [min(int(p * frame_count), frame_count - 1) for p in base_proportions]
    frame_indices.sort()
    
    print(f"Selected frames at indices: {frame_indices}")
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {idx}")
        
        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    
    # Release the video
    cap.release()
    
    # Check if we got enough frames
    if len(frames) < num_frames:
        raise ValueError(f"Could not extract {num_frames} frames from the video")
    
    # Convert to numpy array and normalize
    sequence = np.array(frames, dtype=np.float32) / 255.0
    
    # Add batch dimension (required for the model)
    sequence = np.expand_dims(sequence, axis=0)
    
    return sequence

def run_inference(model, input_data):
    """
    Run inference with the Keras model.
    
    Args:
        model: Keras model
        input_data: Preprocessed input data
        
    Returns:
        output: Model output
        inference_time: Inference time in seconds
    """
    # Run inference
    start_time = time.time()
    output = model.predict(input_data, verbose=0)
    inference_time = time.time() - start_time
    
    return output, inference_time

def test_video(video_path, model, class_names):
    """
    Test the model on a video.
    
    Args:
        video_path: Path to the video file
        model: Keras model
        class_names: List of class names
        
    Returns:
        prediction: Predicted class
        confidence: Confidence score
        inference_time: Inference time in seconds
    """
    try:
        # Preprocess the video
        input_data = preprocess_video(
            video_path, 
            num_frames=10,  # Use 10 frames as that's what we trained with
            target_size=(224, 224)  # Standard resolution
        )
        
        # Run inference
        output, inference_time = run_inference(model, input_data)
        
        # Get prediction
        prediction_idx = np.argmax(output[0])
        confidence = output[0][prediction_idx]
        prediction = class_names[prediction_idx]
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        # Print top 3 predictions
        top_indices = np.argsort(output[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {class_names[idx]}: {output[0][idx]:.4f}")
        
        return prediction, confidence, inference_time
    
    except Exception as e:
        print(f"Error testing video {video_path}: {e}")
        return None, 0.0, 0.0

def display_frames(video_path, num_frames=10, target_size=(224, 224)):
    """
    Display frames from a video that would be used for inference.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to display
        target_size: Target frame size (width, height)
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices using the same logic as inference
        base_proportions = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        frame_indices = [min(int(p * frame_count), frame_count - 1) for p in base_proportions]
        frame_indices.sort()
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {idx}")
                continue
            
            # Convert to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        
        # Release the video
        cap.release()
        
        # Plot frames
        plt.figure(figsize=(15, 3))
        for i, frame in enumerate(frames):
            plt.subplot(1, len(frames), i+1)
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"Frame {i+1}")
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error displaying frames: {e}")

def run_webcam_demo(model, class_names, fps=30, buffer_size=10):
    """
    Run a webcam demo with real-time sign language recognition.
    
    Args:
        model: Keras model
        class_names: List of class names
        fps: Target frames per second
        buffer_size: Number of frames to buffer for prediction
    """
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Create frame buffer
    frame_buffer = []
    
    print("Running webcam demo. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Add to buffer
        frame_buffer.append(processed_frame)
        
        # Keep buffer at the required size
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
        
        # Make prediction when buffer is full
        prediction = "Esperando más frames..."
        confidence = 0.0
        
        if len(frame_buffer) == buffer_size:
            # Use the same frame selection logic as training
            base_proportions = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
            indices = [min(int(p * buffer_size), buffer_size - 1) for p in base_proportions]
            indices.sort()
            
            # Create sequence from selected frames
            sequence = np.array([frame_buffer[i] for i in indices], dtype=np.float32) / 255.0
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Run inference
            try:
                output, _ = run_inference(model, sequence)
                
                # Get prediction
                prediction_idx = np.argmax(output[0])
                confidence = output[0][prediction_idx]
                prediction = class_names[prediction_idx]
            except Exception as e:
                print(f"Error during inference: {e}")
                prediction = "Error"
        
        # Display results on frame
        text = f"{prediction} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display buffer status
        buffer_status = f"Frames: {len(frame_buffer)}/{buffer_size}"
        cv2.putText(frame, buffer_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show frame
        cv2.imshow("Sign Language Recognition", frame)
        
        # Check for exit
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Paths to model files (adjust these to match your file locations)
    model_path = "best_model.h5"  # Asumiendo que está en la carpeta src
    metadata_path = "model_metadata.txt"  # Asumiendo que está en la carpeta src
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found!")
        # Look for h5 files in current directory
        h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if h5_files:
            print(f"Available .h5 files: {h5_files}")
            model_path = h5_files[0]
            print(f"Using {model_path}")
        else:
            print("No .h5 files found in current directory")
            exit(1)
    
    # Load the model
    model = load_keras_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Load class names from metadata or fallback to indices
    class_names = load_class_names(metadata_path) if os.path.exists(metadata_path) else [f"class_{i}" for i in range(13)]
    
    while True:
        print("\nOpciones:")
        print("1. Probar con un video")
        print("2. Ejecutar demo con webcam")
        print("3. Salir")
        
        choice = input("Selecciona una opción (1-3): ")
        
        if choice == '1':
            video_path = input("Ingresa la ruta del video a probar: ")
            if os.path.exists(video_path):
                print("Mostrando frames que serán utilizados:")
                display_frames(video_path)
                print("Ejecutando predicción...")
                test_video(video_path, model, class_names)
            else:
                print(f"ERROR: El archivo {video_path} no existe!")
        
        elif choice == '2':
            print("Iniciando demo con webcam...")
            run_webcam_demo(model, class_names)
        
        elif choice == '3':
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida. Intenta de nuevo.")