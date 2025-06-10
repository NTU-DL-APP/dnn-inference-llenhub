# train.py
# Script to train a Fashion-MNIST CNN model with residual blocks and attention,
# export architecture + weights, and use GPU when available

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dense, Flatten, Dropout, Reshape, Add,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling2D
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
        description="Train a Fashion-MNIST CNN with residual & attention and export .json + .npz files"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)"
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


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    # if shape changes, project shortcut
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', activation=None)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def attention_block(x, num_heads=4, key_dim=32):
    # flatten spatial dims into sequence
    b, h, w, c = x.shape
    seq = tf.reshape(x, (-1, h*w, c))
    # self-attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(seq, seq)
    # add & norm
    seq = Add()([seq, attn])
    seq = LayerNormalization()(seq)
    # restore spatial dims
    x = tf.reshape(seq, (-1, h, w, c))
    return x


def build_model(input_shape=(28, 28, 1), num_classes=10):
    inp = Input(shape=input_shape)
    # initial Conv
    x = Conv2D(32, 3, padding='same', activation=None)(inp)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # residual block
    x = residual_block(x, filters=32)
    x = MaxPooling2D(pool_size=2)(x)

    # deeper block
    x = residual_block(x, filters=64, stride=1)
    x = MaxPooling2D(pool_size=2)(x)

    # attention over feature map
    x = attention_block(x, num_heads=4, key_dim=32)

    # classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model


def load_data():
    # Load Fashion-MNIST and normalize to [0,1]
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    # expand dims for channel
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def main():
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks: reduce LR on plateau + early stopping
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True
        )
    ]

    # Train with augmentation
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
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