import os
import zipfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# MODEL_PATH = "model/euphoscan_model.keras"
MODEL_PATH = "model/euphoscan_render.keras"

print("=" * 60)
print("PYTHON:", os.sys.version)
print("TF:", tf.__version__)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):
    print("MODEL SIZE:", os.path.getsize(MODEL_PATH))
    import hashlib

    with open(MODEL_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    print("SHA256:", sha)

    try:
        z = zipfile.ZipFile(MODEL_PATH)
        print("ZIP CONTENTS:")
        print(z.namelist())
        z.close()
    except Exception as e:
        print("ZIP ERROR:", e)

print("=" * 60)

try:
    print("LOADING MODEL...")
    model = load_model(MODEL_PATH, compile=False)


    print("SUCCESS")
    print("INPUT SHAPE:", model.input_shape)

    for layer in model.layers[:10]:
        print(layer.name, layer.__class__.__name__)

except Exception as e:
    print("MODEL LOAD FAILED")
    print(type(e).__name__)
    print(str(e))