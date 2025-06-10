# train.py
# Script to train a Fashion-MNIST model and export architecture + weights

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Fashion-MNIST model and export .json + .npz files"
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
    # Load Fashion-MNIST and preprocess
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # normalize to [0,1], flatten later in model
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape=(28, 28), num_classes=10):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return model


def main():
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1
    )

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Export architecture
    arch_path = os.path.join(args.model_dir, 'fashion_mnist.json')
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    print(f"Saved model architecture to {arch_path}")

    # Export weights as NPZ
    weights = model.get_weights()
    weights_dict = {f'arr_{i}': w for i, w in enumerate(weights)}
    npz_path = os.path.join(args.model_dir, 'fashion_mnist.npz')
    np.savez(npz_path, **weights_dict)
    print(f"Saved model weights to {npz_path}")

    # Save full model to HDF5
    h5_path = os.path.join(args.model_dir, 'fashion_mnist.h5')
    model.save(h5_path)
    print(f"Saved full model to {h5_path}")



if __name__ == '__main__':
    main()
