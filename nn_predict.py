import numpy as np
import json

# === Activation functions ===
def relu(x):
    # element-wise max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # subtract the row-max for numeric stability,
    # then exponentiate & normalize each row so it sums to 1
    # (x is shape [batch, units])
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg   = layer['config']
        wnames= layer['weights']

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]  # e.g. 'dense_1/kernel:0'
            b = weights[wnames[1]]  # e.g. 'dense_1/bias:0'
            x = dense(x, W, b)
            act = cfg.get("activation")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)

    return x

# You can leave nn_inference() as a call-through
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

