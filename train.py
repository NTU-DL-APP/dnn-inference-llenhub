# train.py
# Script to train a Fashion-MNIST CNN model and export architecture + weights
# Added GPU setup for TensorFlow

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, Reshape
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

# === GPU Configuration ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, defaulting to CPU.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Fashion-MNIST CNN model and export .json + .npz files"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for training (default: 128)"
    )
    parser.add_argument(
        "--model_dir", type=str, default="model",
        help="Directory to save JSON + NPZ (default: ./model)"
    )
    return parser.parse_args()


def load_data():
    # Load Fashion-MNIST and normalize to [0,1]
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    # expand dims for channel
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Reshape(input_shape, input_shape=input_shape),
        Conv2D(32, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(64, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    return model


def main():
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with augmentation
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        validation_data=(x_test, y_test),
        epochs=args.epochs
    )

    # Evaluate on test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Export architecture
    arch_path = os.path.join(args.model_dir, 'fashion_mnist.json')
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    print(f"Saved model architecture to {arch_path}")

    # Export weights using layer weight names
    weights_dict = {}
    for layer in model.layers:
        for weight_tensor in layer.weights:
            weights_dict[weight_tensor.name] = weight_tensor.numpy()
    npz_path = os.path.join(args.model_dir, 'fashion_mnist.npz')
    np.savez(npz_path, **weights_dict)
    print(f"Saved model weights to {npz_path}")

    # Save full model to HDF5
    h5_path = os.path.join(args.model_dir, 'fashion_mnist.h5')
    model.save(h5_path)
    print(f"Saved full model to {h5_path}")


if __name__ == '__main__':
    main()
