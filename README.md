# Pattern Recognition Lab – Image Classifier

This project trains a convolutional neural network (CNN) to classify custom images (astronaut, flower, cube variations, guillotine, etc.) using TensorFlow/Keras.

## Project Layout

- `main.py` – end-to-end training, evaluation, confusion matrix, and sample predictions.
- `training/<class_name>/` – training images grouped by class folders.
- `testing/<class_name>/` – testing images grouped by class folders (same class names as training).
- `.venv/` – Python virtual environment (optional; you can recreate it).
- `requirements.txt` – exact Python dependencies.

## Prerequisites

- **macOS**: macOS 12+ with Python 3.9+ (project currently uses 3.13). Apple Silicon automatically uses Metal for GPU acceleration when running the arm64 TensorFlow build.
- **Windows**: Windows 10/11 with Python 3.9+ (64-bit). For NVIDIA GPU support you need CUDA 12+ and cuDNN 9+, otherwise TensorFlow runs on CPU.

## Setup Instructions

1. **Clone / copy** the repository wherever you want.
2. **Create & activate a virtual environment** (recommended):
   - **macOS / Linux**:
     ```bash
     cd /path/to/lab-sub
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     cd C:\path\to\lab-sub
     python -m venv .venv
     .\.venv\Scripts\Activate
     ```
3. **Install dependencies** (same commands on both platforms once the venv is active):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

Activate the virtual environment (if not already active) and run:

- **macOS / Linux**:
  ```bash
  python main.py
  ```
- **Windows (PowerShell)**:
  ```powershell
  python main.py
  ```

The script will:
1. Load training/testing datasets using `ImageDataGenerator`.
2. Train the CNN for 15 epochs (adjust `epochs` in `main.py` if needed).
3. Plot training/validation accuracy & loss.
4. Display a confusion matrix and classification report on the testing set.
5. Show random test images with predicted vs actual labels.

> macOS note: `plt.show()` blocks until you close the figure windows. Close each plot window to let the script continue. If you need headless execution, replace `plt.show()` calls with `plt.savefig(...)` or set `matplotlib.use("Agg")`.
>
> Windows tip: if PowerShell execution policy blocks the venv activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before `.\.venv\Scripts\Activate`.

## Common Tasks

- **Check GPU availability**:
  ```bash
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```
- **Regenerate `requirements.txt` after adding packages**:
  ```bash
  python -m pip freeze > requirements.txt
  ```
- **Run without the virtual environment**: install dependencies globally (`pip install -r requirements.txt`), then run `python main.py`.

## Troubleshooting

- **`ModuleNotFoundError`**: ensure the virtual env is active and dependencies installed.
- **`FileNotFoundError: 'training/'` or `'testing/'`**: verify folder names and paths relative to the project root.
- **Matplotlib windows block execution**: close them manually or switch to saving plots.

Happy training!

