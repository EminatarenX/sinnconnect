import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure TensorFlow to use the Metal API on Apple Silicon
try:
    # For M-series Macs
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using Apple Metal GPU acceleration")
    else:
        print("No GPU found, using CPU")
except Exception as e:
    print(f"Error configuring GPU: {e}")

def load_dataset(dataset_dir, img_size=(224, 224), n_sequences=5, n_frames=10):
    """
    Load the dataset from the directory structure.
    
    Args:
        dataset_dir: Path to the dataset directory
        img_size: Image size (width, height)
        n_sequences: Number of sequences per video
        n_frames: Number of frames per sequence
        
    Returns:
        X: Sequences data with shape (n_samples, n_frames, height, width, channels)
        y: Labels
        class_names: List of class names
    """
    class_names = os.listdir(dataset_dir)
    class_names = [d for d in class_names if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()  # Sort to ensure consistent indexing
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Loading classes")):
        class_dir = os.path.join(dataset_dir, class_name)
        
        # Get all sets (videos)
        sets = os.listdir(class_dir)
        sets = [s for s in sets if os.path.isdir(os.path.join(class_dir, s))]
        
        for set_name in tqdm(sets, desc=f"Loading {class_name} sets", leave=False):
            set_dir = os.path.join(class_dir, set_name)
            
            # Get all sequences in this set
            sequences = os.listdir(set_dir)
            sequences = [s for s in sequences if os.path.isdir(os.path.join(set_dir, s))]
            
            for seq_name in sequences:
                seq_dir = os.path.join(set_dir, seq_name)
                
                # Load all frames in this sequence
                frames = []
                frame_files = [f for f in os.listdir(seq_dir) if f.endswith('.jpg')]
                frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by frame number
                
                for frame_file in frame_files:
                    frame_path = os.path.join(seq_dir, frame_file)
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    img = cv2.resize(img, img_size)
                    frames.append(img)
                
                # Ensure we have exactly n_frames
                if len(frames) == n_frames:
                    X.append(frames)
                    y.append(class_idx)
    
    # Convert lists to numpy arrays
    X = np.array(X, dtype=np.float32) / 255.0  # Normalize pixel values
    y = tf.keras.utils.to_categorical(y)  # One-hot encode labels
    
    print(f"Dataset loaded: {len(X)} samples, {len(class_names)} classes")
    return X, y, class_names

def create_cnn_lstm_model(input_shape, num_classes, mobile_net=True):
    """
    Create a CNN-LSTM model for sign language recognition.
    
    Args:
        input_shape: Shape of input sequences (frames, height, width, channels)
        num_classes: Number of classes to predict
        mobile_net: Whether to use MobileNetV2 as the CNN base (more efficient for mobile)
        
    Returns:
        model: Compiled Keras model
    """
    # Create the model
    model = Sequential()
    
    if mobile_net:
        # Using MobileNetV2 as base CNN (more efficient for mobile devices)
        base_model = MobileNetV2(
            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model to prevent training its weights (transfer learning)
        base_model.trainable = False
        
        # Wrap the CNN in TimeDistributed layer to process each frame
        model.add(TimeDistributed(base_model, input_shape=input_shape))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(256, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
    else:
        # Simple custom CNN
        model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(256, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
    
    # LSTM layers to process the sequence
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(X, y, class_names, mobile_net=True, epochs=50, batch_size=32):
    """
    Train and evaluate the model, save the best model weights.
    
    Args:
        X: Input sequences data
        y: Target labels (one-hot encoded)
        class_names: List of class names
        mobile_net: Whether to use MobileNetV2 as CNN base
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        history: Training history
        model: Trained model
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create the model
    model = create_cnn_lstm_model(X.shape[1:], len(class_names), mobile_net)
    model.summary()
    
    # Callbacks for training
    checkpoint = ModelCheckpoint(
        'best_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return history, model

def convert_to_tflite(model, output_path='model.tflite'):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save the TFLite model
    """
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted to TFLite and saved to {output_path}")
    
    # Also save class names for reference in the mobile app
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

def save_model_metadata(model, class_names):
    """
    Save model metadata for use in the mobile app.
    
    Args:
        model: Trained model
        class_names: List of class names
    """
    # Save model input shape
    input_shape = model.input_shape[1:]  # Skip batch dimension
    
    metadata = {
        "input_shape": input_shape,
        "num_frames": input_shape[0],
        "frame_height": input_shape[1],
        "frame_width": input_shape[2],
        "channels": input_shape[3],
        "class_names": class_names
    }
    
    # Save as a simple text file
    with open('model_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model metadata saved to model_metadata.txt")

if __name__ == "__main__":
    # Load the dataset
    dataset_dir = "../data/dataset"
    X, y, class_names = load_dataset(dataset_dir)
    
    # Train and evaluate the model
    history, model = train_and_evaluate(X, y, class_names, mobile_net=True)
    
    # Save model metadata
    save_model_metadata(model, class_names)
    
    # Convert model to TFLite
    convert_to_tflite(model)
    
    print("Training and conversion complete!")