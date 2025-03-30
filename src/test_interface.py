import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def load_tflite_model(model_path):
    """
    Load a TensorFlow Lite model.
    
    Args:
        model_path: Path to the TFLite model
        
    Returns:
        interpreter: TFLite interpreter
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model loaded successfully")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    return interpreter, input_details, output_details

def load_class_names(file_path):
    """
    Load class names from a text file.
    
    Args:
        file_path: Path to the class names file
        
    Returns:
        class_names: List of class names
    """
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names

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
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
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

def run_inference(interpreter, input_details, output_details, input_data):
    """
    Run inference with the TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        input_data: Preprocessed input data
        
    Returns:
        output: Model output
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output, inference_time

def test_video(video_path, interpreter, input_details, output_details, class_names):
    """
    Test the model on a video.
    
    Args:
        video_path: Path to the video file
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
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
            num_frames=input_details[0]['shape'][1],
            target_size=(input_details[0]['shape'][2], input_details[0]['shape'][3])
        )
        
        # Run inference
        output, inference_time = run_inference(interpreter, input_details, output_details, input_data)
        
        # Get prediction
        prediction_idx = np.argmax(output[0])
        confidence = output[0][prediction_idx]
        prediction = class_names[prediction_idx]
        
        return prediction, confidence, inference_time
    
    except Exception as e:
        print(f"Error testing video {video_path}: {e}")
        return None, 0.0, 0.0

def display_frames(video_path, num_frames=10, target_size=(224, 224)):
    """
    Display frames from a video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to display
        target_size: Target frame size (width, height)
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    
    # Release the video
    cap.release()
    
    # Plot frames
    plt.figure(figsize=(15, 3))
    for i, frame in enumerate(frames):
        plt.subplot(1, num_frames, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {i+1}")
    
    plt.tight_layout()
    plt.show()

def evaluate_test_set(test_dir, interpreter, input_details, output_details, class_names):
    """
    Evaluate the model on a test set.
    
    Args:
        test_dir: Path to the test directory
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        class_names: List of class names
    """
    # Dictionary to store results
    class_results = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    inference_times = []
    
    # Test each class
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        
        # Skip non-directories
        if not os.path.isdir(class_dir) or class_name not in class_names:
            continue
        
        # Get all video files
        video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"\nTesting class: {class_name}")
        for video_file in tqdm(video_files, desc=f"Testing {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            
            # Test the video
            prediction, confidence, inf_time = test_video(
                video_path, interpreter, input_details, output_details, class_names
            )
            
            if prediction is not None:
                # Update results
                class_results[class_name]['total'] += 1
                if prediction == class_name:
                    class_results[class_name]['correct'] += 1
                
                inference_times.append(inf_time)
    
    # Calculate accuracy per class
    class_accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for class_name, results in class_results.items():
        if results['total'] > 0:
            accuracy = results['correct'] / results['total']
            class_accuracies[class_name] = accuracy
            
            total_correct += results['correct']
            total_samples += results['total']
            
            print(f"{class_name}: {accuracy:.4f} ({results['correct']}/{results['total']})")
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    
    # Plot class accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.show()

def run_webcam_demo(interpreter, input_details, output_details, class_names, fps=30, buffer_size=10):
    """
    Run a webcam demo with real-time sign language recognition.
    
    Args:
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        class_names: List of class names
        fps: Target frames per second
        buffer_size: Number of frames to buffer for prediction
    """
    # Get input shape details
    input_shape = input_details[0]['shape']
    num_frames = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    
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
        processed_frame = cv2.resize(frame, (width, height))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Add to buffer
        frame_buffer.append(processed_frame)
        
        # Keep buffer at the required size
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
        
        # Make prediction when buffer is full
        prediction = "Waiting for more frames..."
        confidence = 0.0
        
        if len(frame_buffer) == buffer_size:
            # Sample frames from buffer to match model's expected num_frames
            indices = np.linspace(0, buffer_size-1, num_frames, dtype=int)
            sequence = np.array([frame_buffer[i] for i in indices], dtype=np.float32) / 255.0
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Run inference
            output, _ = run_inference(interpreter, input_details, output_details, sequence)
            
            # Get prediction
            prediction_idx = np.argmax(output[0])
            confidence = output[0][prediction_idx]
            prediction = class_names[prediction_idx]
        
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
    # Load the model
    model_path = "../models/model.tflite"
    interpreter, input_details, output_details = load_tflite_model(model_path)
    
    # Load class names
    class_names = load_class_names("class_names.txt")
    
    # Run tests on a directory of test videos (optional)
    # Uncomment the line below to run evaluation on a test set
    # evaluate_test_set("test_videos", interpreter, input_details, output_details, class_names)
    
    # Run webcam demo
    run_webcam_demo(interpreter, input_details, output_details, class_names)