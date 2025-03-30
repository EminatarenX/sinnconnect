import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm

def convert_to_tflite(model_path, output_path='model.tflite', optimization_level='default'):
    """
    Convert Keras model to TensorFlow Lite with optimization.
    
    Args:
        model_path: Path to the saved Keras model (.h5)
        output_path: Path to save the TFLite model
        optimization_level: Optimization level ('default', 'float16', 'dynamic_range', 'full_integer')
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization based on the specified level
    if optimization_level == 'default':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Using DEFAULT optimization")
    
    elif optimization_level == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Using FLOAT16 optimization")
    
    elif optimization_level == 'dynamic_range':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Using DYNAMIC RANGE quantization")
    
    elif optimization_level == 'full_integer':
        # For full integer quantization, you need representative dataset
        def representative_dataset_gen():
            # Generate random data that matches your input shape
            # This is a placeholder - ideally you should use real data
            input_shape = model.input_shape
            for _ in range(100):  # Generate 100 samples
                dummy_input = np.random.rand(*input_shape)
                yield [dummy_input.astype(np.float32)]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("Using FULL INTEGER quantization")
    
    else:
        raise ValueError(f"Unknown optimization level: {optimization_level}")
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Model converted and saved to {output_path} ({size_kb:.2f} KB)")
    
    return output_path

def benchmark_tflite_model(model_path, input_shape, num_runs=50):
    """
    Benchmark a TFLite model for inference speed.
    
    Args:
        model_path: Path to the TFLite model
        input_shape: Shape of input data (including batch dimension)
        num_runs: Number of inference runs for averaging
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create random input data
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # Warm-up runs
    print("Warming up...")
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Benchmark runs
    print(f"Running benchmark ({num_runs} iterations)...")
    inference_times = []
    
    for _ in tqdm(range(num_runs)):
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Measure inference time
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        inference_times.append(inference_time)
    
    # Calculate statistics
    avg_time = np.mean(inference_times) * 1000  # Convert to ms
    std_time = np.std(inference_times) * 1000
    min_time = np.min(inference_times) * 1000
    max_time = np.max(inference_times) * 1000
    
    print(f"\nBenchmark results for {model_path}:")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Standard deviation: {std_time:.2f} ms")
    print(f"  Min inference time: {min_time:.2f} ms")
    print(f"  Max inference time: {max_time:.2f} ms")
    
    return avg_time

def compare_optimizations(model_path, input_shape):
    """
    Compare different optimization techniques for TFLite conversion.
    
    Args:
        model_path: Path to the saved Keras model (.h5)
        input_shape: Shape of input data (including batch dimension)
    """
    optimization_levels = [
        'default',
        'float16',
        'dynamic_range',
        'full_integer'
    ]
    
    results = {}
    
    for level in optimization_levels:
        print(f"\n--- Testing {level.upper()} optimization ---")
        output_path = f"model_{level}.tflite"
        
        # Convert model with this optimization
        try:
            tflite_path = convert_to_tflite(model_path, output_path, level)
            
            # Benchmark the model
            avg_time = benchmark_tflite_model(tflite_path, input_shape)
            
            # Get file size
            size_kb = os.path.getsize(tflite_path) / 1024
            
            results[level] = {
                'path': tflite_path,
                'avg_time': avg_time,
                'size_kb': size_kb
            }
        
        except Exception as e:
            print(f"Error with {level} optimization: {e}")
    
    # Print comparison results
    print("\n=== Optimization Comparison ===")
    print(f"{'Optimization':<15} {'Size (KB)':<15} {'Avg Time (ms)':<15}")
    print("-" * 45)
    
    for level, data in results.items():
        print(f"{level:<15} {data['size_kb']:<15.2f} {data['avg_time']:<15.2f}")
    
    # Recommend the best option
    if results:
        # Find the fastest model
        fastest_level = min(results.items(), key=lambda x: x[1]['avg_time'])[0]
        
        # Find the smallest model
        smallest_level = min(results.items(), key=lambda x: x[1]['size_kb'])[0]
        
        print("\nRecommendations:")
        print(f"  Fastest model: {fastest_level} ({results[fastest_level]['avg_time']:.2f} ms)")
        print(f"  Smallest model: {smallest_level} ({results[smallest_level]['size_kb']:.2f} KB)")
        
        # Balanced recommendation (using harmonic mean of normalized metrics)
        normalized_times = {k: v['avg_time'] / max(r['avg_time'] for r in results.values()) 
                           for k, v in results.items()}
        normalized_sizes = {k: v['size_kb'] / max(r['size_kb'] for r in results.values()) 
                           for k, v in results.items()}
        
        balanced_scores = {k: 2 / (normalized_times[k] + normalized_sizes[k]) 
                          for k in results.keys()}
        
        balanced_recommendation = max(balanced_scores.items(), key=lambda x: x[1])[0]
        
        print(f"  Best balance of size and speed: {balanced_recommendation}")
        print(f"    Size: {results[balanced_recommendation]['size_kb']:.2f} KB")
        print(f"    Time: {results[balanced_recommendation]['avg_time']:.2f} ms")
        
        # Copy the recommended model to the final name
        recommended_path = results[balanced_recommendation]['path']
        final_path = "model_optimized.tflite"
        
        with open(recommended_path, 'rb') as src, open(final_path, 'wb') as dst:
            dst.write(src.read())
        
        print(f"\nRecommended model copied to {final_path}")

if __name__ == "__main__":
    # Path to the saved Keras model
    model_path = "./best_model.h5"
    
    # Input shape should match your model's input shape
    # Format is (batch_size, num_frames, height, width, channels)
    # Example for 1 batch, 10 frames of 224x224 RGB images:
    input_shape = (1, 10, 224, 224, 3)
    
    # Compare different optimization techniques
    compare_optimizations(model_path, input_shape)