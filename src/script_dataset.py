import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_videos(input_folder, output_folder, target_size=(224, 224)):
    """
    Normalize videos by resizing them to a target size while preserving orientation.
    
    Args:
        input_folder: Path to folder containing the original videos
        output_folder: Path to store normalized videos
        target_size: Target resolution (width, height)
    """
    create_directory(output_folder)
    
    # Get all video files
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]
    
    for video_file in tqdm(video_files, desc="Normalizing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        
        # Skip if already normalized
        if os.path.exists(output_path):
            continue
        
        # Open video - explicitly disable rotation flags
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # Disable auto-orientation
        
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            continue
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video {video_file}: Original size {width}x{height}, Target size {target_size}")
        
        # Create VideoWriter object for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame while preserving orientation - no rotation
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            
            # Write frame to output video
            out.write(resized_frame)
        
        # Release resources
        cap.release()
        out.release()
        
    print(f"Normalized {len(video_files)} videos to {target_size}")

def extract_sequences(normalized_folder, output_base_folder, n_sequences=5, n_frames=10):
    """
    Extract sequences of frames from normalized videos using strategic frame selection.
    
    Args:
        normalized_folder: Path to folder containing normalized videos
        output_base_folder: Base path to store extracted sequences
        n_sequences: Number of sequences to extract per video
        n_frames: Number of frames per sequence
    """
    create_directory(output_base_folder)
    
    # Get all video files
    video_files = [f for f in os.listdir(normalized_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]
    
    for idx, video_file in enumerate(tqdm(video_files, desc="Extracting sequences")):
        video_path = os.path.join(normalized_folder, video_file)
        
        # Create directory for this video's sequences
        video_seq_folder = os.path.join(output_base_folder, f"set_{idx+1}")
        
        # Skip if already processed
        if os.path.exists(video_seq_folder) and len(os.listdir(video_seq_folder)) >= n_sequences:
            continue
        
        create_directory(video_seq_folder)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        
        print(f"Video {video_file}: {frame_count} frames, {fps} FPS, duration: {duration:.2f}s")
        
        if frame_count < 15:  # Need minimum frames
            print(f"Warning: Video {video_file} has too few frames ({frame_count})")
            continue
            
        # Read all frames at once to avoid seeking issues
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
            
        if len(all_frames) == 0:
            print(f"No se pudieron leer frames del video {video_file}")
            continue
            
        # Following the user's idea for strategic frame selection
        # For a 3-second video at 30fps (90 frames), we'll use patterns like:
        # Seq 1: 1, 6, 18, 24, 30, 36, 42, 48, 54, 60
        # Seq 2: 3, 8, 20, 26, 32, 38, 44, 50, 56, 62
        # But we'll adapt for the actual frame count
        
        # Define base reference positions (0-indexed)
        # These are the proportions along the video length
        base_proportions = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        
        for seq_idx in range(n_sequences):
            seq_folder = os.path.join(video_seq_folder, f"seq_{seq_idx+1}")
            create_directory(seq_folder)
            
            # Add an offset for each sequence (shift by 2% for each sequence)
            offset = (seq_idx * 0.02) % 0.1  # Cycle through 0, 0.02, 0.04, 0.06, 0.08 for variety
            
            # Calculate actual frame indices with offset
            sequence_proportions = [(p + offset) % 1.0 for p in base_proportions]
            frame_indices = [min(int(p * frame_count), frame_count - 1) for p in sequence_proportions]
            
            # Sort to ensure frames are in chronological order
            frame_indices.sort()
            
            print(f"Secuencia {seq_idx+1}: Frames {frame_indices}")
            
            # Extract and save the frames
            for frame_idx, frame_pos in enumerate(frame_indices):
                if frame_pos < len(all_frames):
                    frame = all_frames[frame_pos]
                    
                    # Save frame as JPG
                    frame_path = os.path.join(seq_folder, f"frame_{frame_idx+1:02d}.jpg")
                    cv2.imwrite(frame_path, frame)
                else:
                    print(f"Warning: Frame position {frame_pos} out of range (max: {len(all_frames)-1})")
        
        # Release resources
        cap.release()
    
    print(f"Extracted {n_sequences} sequences from {len(video_files)} videos")

def process_dataset(base_folder, target_size=(224, 224), n_sequences=5, n_frames=10):
    """
    Process the entire dataset.
    
    Args:
        base_folder: Path to the base folder (no_normalized)
        target_size: Target resolution (width, height)
        n_sequences: Number of sequences to extract per video
        n_frames: Number of frames per sequence
    """
    # Create organized dataset folder
    dataset_folder = os.path.join(os.path.dirname(base_folder), "dataset")
    create_directory(dataset_folder)
    print(f"Dataset será creado en: {dataset_folder}")
    
    # Process each class
    for class_name in tqdm(os.listdir(base_folder), desc="Processing classes"):
        class_folder = os.path.join(base_folder, class_name)
        
        # Skip non-directories
        if not os.path.isdir(class_folder):
            print(f"Saltando {class_name}: No es un directorio")
            continue
        
        print(f"\n{'='*50}")
        print(f"Procesando clase: {class_name}")
        
        # Check for videos in class folder
        video_files = [f for f in os.listdir(class_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]
        if len(video_files) == 0:
            print(f"ADVERTENCIA: No se encontraron videos en {class_folder}")
            continue
        
        print(f"Encontrados {len(video_files)} videos para la clase {class_name}")
        
        # Create normalized videos folder
        normalized_folder = os.path.join(class_folder, "videos_normalizados")
        print(f"Normalizando videos a {normalized_folder}")
        
        # Normalize videos - We're looking for .mp4 files directly in the class folder
        normalize_videos(class_folder, normalized_folder, target_size)
        
        # Verify normalized videos were created
        if not os.path.exists(normalized_folder) or len(os.listdir(normalized_folder)) == 0:
            print(f"ERROR: No se pudieron crear videos normalizados en {normalized_folder}")
            continue
        
        # Create class folder in the dataset
        class_dataset_folder = os.path.join(dataset_folder, class_name)
        print(f"Extrayendo secuencias a {class_dataset_folder}")
        
        # Extract sequences
        extract_sequences(normalized_folder, class_dataset_folder, n_sequences, n_frames)
        
        # Verify sequences were created
        if os.path.exists(class_dataset_folder):
            set_folders = [d for d in os.listdir(class_dataset_folder) if os.path.isdir(os.path.join(class_dataset_folder, d))]
            if len(set_folders) > 0:
                # Check first set folder
                first_set = os.path.join(class_dataset_folder, set_folders[0])
                seq_folders = [d for d in os.listdir(first_set) if os.path.isdir(os.path.join(first_set, d))]
                if len(seq_folders) > 0:
                    # Check first sequence folder
                    first_seq = os.path.join(first_set, seq_folders[0])
                    frames = [f for f in os.listdir(first_seq) if f.endswith('.jpg')]
                    print(f"Ejemplo: Set {set_folders[0]}, Secuencia {seq_folders[0]}, {len(frames)} frames")
    
    print("\n" + "="*50)
    print("¡Procesamiento del dataset completado!")
    print(f"Dataset guardado en: {dataset_folder}")
    
    # Check final dataset structure
    class_folders = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    print(f"Clases procesadas: {len(class_folders)}")
    for class_name in class_folders:
        class_path = os.path.join(dataset_folder, class_name)
        set_count = len([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])
        print(f"  - {class_name}: {set_count} sets")

if __name__ == "__main__":
    # Base folder containing the unnormalized videos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(os.path.dirname(current_dir), "data", "no_normalized")
    print(f"Buscando carpeta en: {base_folder}")
    
    print(f"Buscando videos en: {os.path.abspath(base_folder)}")
    
    # Check if folder exists
    if not os.path.exists(base_folder):
        print(f"ERROR: La carpeta {base_folder} no existe. Verifica la ruta.")
        exit(1)
    
    # List classes
    classes = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    if len(classes) == 0:
        print(f"WARNING: No se encontraron carpetas de clases en {base_folder}")
    else:
        print(f"Clases encontradas: {classes}")
        
        # List videos in first class as sample
        sample_class = classes[0]
        sample_path = os.path.join(base_folder, sample_class)
        videos = [f for f in os.listdir(sample_path) if f.endswith(('.mp4', '.MOV', '.mov'))]
        print(f"Videos en {sample_class}: {len(videos)}")
        
        if len(videos) > 0:
            print(f"Ejemplo de video: {videos[0]}")
    
    # Process the dataset
    process_dataset(base_folder)