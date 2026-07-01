# export_tflite.py — TFLite export for edge deployment — SR
# converts PyTorch model → ONNX → TFLite (or via TF)

import torch
import numpy as np
import os
from typing import Optional, Tuple


def export_to_onnx(model, input_shape: Tuple, save_path: str,
                   device=None):
    """Export PyTorch model to ONNX format.

    Args:
        model: trained PyTorch model
        input_shape: (batch_size, seq_len, features)
        save_path: path to save .onnx file
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)

    try:
        torch.onnx.export(
            model, dummy_input, save_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=11,
        )
        print(f"Exported ONNX to {save_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        # FIXME: some models fail ONNX export — fallback to TF
        return False


def export_to_tflite_via_tf(model, input_shape: Tuple,
                            save_path: str, quantize: bool = True):
    """Export via PyTorch → TF → TFLite (alternative path).

    Uses onnx-tf or manual weight transfer.
    For simplicity, we implement a direct PyTorch → TFLite path
    using torch's built-in export or manual TF model construction.

    # NOTE: this is a simplified version
    # full pipeline would use onnx2tf or similar
    """
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        # load ONNX model
        onnx_path = save_path.replace('.tflite', '.onnx')
        if os.path.exists(onnx_path):
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(save_path.replace('.tflite', '_tf'))

            # convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(
                save_path.replace('.tflite', '_tf')
            )
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_model = converter.convert()
            with open(save_path, 'wb') as f:
                f.write(tflite_model)

            print(f"Exported TFLite to {save_path} ({len(tflite_model) / 1024:.1f} KB)")
            return True

    except ImportError:
        print("onnx-tf not available. Using alternative export path.")
        return export_tflite_simple(model, input_shape, save_path, quantize)

    return False


def export_tflite_simple(model, input_shape: Tuple,
                         save_path: str, quantize: bool = True):
    """Simple TFLite export via numpy weight extraction + TF reconstruction.

    This is a fallback for when ONNX pipeline fails.
    Reconstructs a minimal TF model with same weights.
    """
    try:
        import tensorflow as tf

        # extract weights from PyTorch model
        model.eval()
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()

        # build equivalent TF model
        # for GRU: use tf.keras.layers.GRU
        # this is model-specific — implement per architecture
        print("Fallback: building TF model from PyTorch weights...")
        print(f"  Model has {len(weights)} weight tensors")

        # TODO: implement full weight transfer for each model type
        # for now, save weights dict for manual reconstruction
        np.savez(save_path.replace('.tflite', '_weights.npz'), **weights)
        print(f"Saved weights to {save_path.replace('.tflite', '_weights.npz')}")

        return False

    except Exception as e:
        print(f"TFLite export failed: {e}")
        return False


def measure_tflite_latency(tflite_path: str, input_shape: Tuple,
                           n_runs: int = 100) -> dict:
    """Measure TFLite inference latency."""
    try:
        import tensorflow as tf
        import time

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # generate random input
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

        # measure
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        latencies = np.array(latencies)

        return {
            'latency_ms_mean': float(np.mean(latencies)),
            'latency_ms_std': float(np.std(latencies)),
            'latency_ms_p50': float(np.percentile(latencies, 50)),
            'model_size_bytes': os.path.getsize(tflite_path),
            'model_size_kb': os.path.getsize(tflite_path) / 1024,
        }

    except ImportError:
        print("TensorFlow not available for latency measurement")
        return {}


if __name__ == "__main__":
    # test export pipeline
    from src.model import build_model, count_params
    from src.utils import set_seed

    set_seed(42)

    input_size = 5
    seq_len = 12
    batch_size = 1

    for model_name in ['gru', 'lstm', 'rnn', 'tcn', 'ltransformer']:
        model = build_model(model_name, input_size)
        print(f"\n{model_name}: {count_params(model)} params")

        # try ONNX export
        onnx_path = f"runs/{model_name}.onnx"
        os.makedirs("runs", exist_ok=True)
        export_to_onnx(model, (batch_size, seq_len, input_size), onnx_path)
