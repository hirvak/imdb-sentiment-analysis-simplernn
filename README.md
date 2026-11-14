# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

An end-to-end Natural Language Processing (NLP) project that classifies IMDB movie reviews into **Positive** or **Negative** sentiments using a Recurrent Neural Network (SimpleRNN) and serves real-time predictions through an interactive Streamlit web application.

---

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Model Architecture & Training](#-model-architecture--training)
- [How It Works (Pipeline)](#-how-it-works-pipeline)
- [Installation](#-installation)
- [Usage Instructions](#-usage-instructions)
- [Troubleshooting & Known Issues](#-troubleshooting--known-issues)
- [Future Improvements](#-future-improvements)
- [License & Author](#-license--author)

---

## 📖 Overview

Sentiment analysis plays a critical role in understanding user opinions, feedback, and customer reviews. This project explores sequential deep learning by implementing a **Simple Recurrent Neural Network (SimpleRNN)** trained on the benchmark **IMDB Movie Reviews Dataset**.

Unlike standard feed-forward networks, recurrent architectures maintain an internal memory state across sequence steps, making them suited for processing sequential natural language data. The trained model is deployed with an intuitive **Streamlit** web app that allows users to type or paste any movie review, producing an immediate sentiment classification with a prediction score and confidence percentage.

---

## ✨ Key Features

- **Interactive Web App**: User-friendly Streamlit interface for instant review evaluation.
- **Real-Time Sentiment Classification**: Accurately classifies text as **Positive** or **Negative**.
- **Probability & Confidence Scoring**: Displays both raw model output (0.0 to 1.0) and a human-readable confidence percentage.
- **Robust Preprocessing Pipeline**: Tokenization, vocabulary index lookup with out-of-vocabulary (OOV) handling, and sequence padding up to 500 tokens.
- **Early Stopping Optimization**: Trained with validation loss monitoring to prevent overfitting and restore the best model weights.
- **Cross-Version Compatibility Utility**: Includes `fix_model.py` to fix HDF5 metadata issues across modern TensorFlow/Keras versions.
- **Modular Notebooks**: Clean Jupyter notebooks detailing embedding exploration, model training, and standalone prediction testing.

---

## 🖥️ Demo

```
+-----------------------------------------------------------+
|         IMDB Movie Review Sentiment Analysis              |
|                                                           |
|  Enter a movie review below to predict its sentiment      |
|  (Positive or Negative).                                  |
|                                                           |
|  [ The movie was fantastic! I really enjoyed it.       ]  |
|                                                           |
|  [ Predict Sentiment ]                                    |
|                                                           |
|  Prediction Result                                        |
|  +-----------------------------------------------------+  |
|  | Sentiment: Positive                                 |  |
|  +-----------------------------------------------------+  |
|  Prediction Score: 0.8093                                 |
|  Confidence: 80.93%                                       |
+-----------------------------------------------------------+
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow 2.x, Keras (Sequential API, Embedding, SimpleRNN, Dense)
- **Web Framework**: Streamlit
- **Data & Sequence Processing**: NumPy, Keras Preprocessing (`text_to_word_sequence`, `pad_sequences`)
- **Model Storage**: HDF5 (`h5py`)

---

## 📂 Project Structure

```text
SimpleRNN/
│
├── main.py                          # Streamlit web application
├── simplernn.ipynb                  # Training pipeline, validation & model export
├── embedding.ipynb                  # Experiments with Keras Embedding layer
├── prediction.ipynb                 # Standalone script/notebook for inference testing
├── fix_model.py                     # Compatibility script fixing HDF5 time_major attribute
│
├── simple_rnn_imdb.h5               # Original trained model in HDF5 format
├── simple_rnn_imdb_backup.h5        # Backup of the original trained model
├── simple_rnn_imdb_compatible.h5   # Patched model compatible with newer Keras versions
│
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

### File Breakdown

- **`main.py`**: The entry point for the Streamlit web application. Caches model loading, processes user input text, runs inference, and renders the result.
- **`simplernn.ipynb`**: The primary training notebook. Downloads IMDB data, builds the Sequential SimpleRNN model, applies EarlyStopping, and trains for sentiment classification.
- **`embedding.ipynb`**: Demonstrates the concept of word representations, integer encoding via `one_hot`, padding sequences, and dense embedding vectors.
- **`prediction.ipynb`**: Interactive notebook testing tokenization, index shifting, and prediction output on sample custom strings.
- **`fix_model.py`**: A utility script addressing Keras 3 / TensorFlow deserialization mismatches by stripping the deprecated `time_major` parameter from the `.h5` config.
- **`simple_rnn_imdb_compatible.h5`**: The production-ready model artifact loaded by `main.py`.

---

## 🧠 Model Architecture & Training

The model is built using Keras's `Sequential` API. It transforms discrete word tokens into dense embedding vectors, passes them through a recurrent layer, and outputs a binary classification probability.

```
Input Tokens (Length: 500)
         │
         ▼
┌─────────────────────────────────┐
│        Embedding Layer          │  input_dim=10000, output_dim=128
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│        SimpleRNN Layer          │  units=128, activation='relu', dropout=0.2
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Dense Output Layer         │  units=1, activation='sigmoid'
└─────────────────────────────────┘
         │
         ▼
Predicted Probability [0.0 - 1.0]
```

### Layer & Parameter Summary

| Layer | Type | Output Shape | Parameters | Details |
| :--- | :--- | :--- | :--- | :--- |
| `embedding_1` | `Embedding` | `(None, 500, 128)` | `1,280,000` | Vocab: 10,000; Dim: 128; Max Length: 500 |
| `simple_rnn_1` | `SimpleRNN` | `(None, 128)` | `32,896` | 128 units, ReLU activation, 20% dropout |
| `dense_1` | `Dense` | `(None, 1)` | `129` | 1 unit, Sigmoid activation |
| **Total** | | | **1,313,025** (~5.01 MB) | **All parameters trainable** |

### Hyperparameters & Training Settings

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dataset** | IMDB Movie Reviews | 50,000 reviews (25,000 train, 25,000 test) |
| **Vocabulary Size** | 10,000 | Top 10,000 most frequent words retained |
| **Max Sequence Length** | 500 | Sequences padded or truncated to 500 tokens |
| **Padding / Truncating** | `'pre'` / `'pre'` | Padded or truncated at the start of sequences |
| **Embedding Dimension** | 128 | Continuous vector representation per word |
| **RNN Hidden Units** | 128 | Simple recurrent memory cells |
| **RNN Activation** | `relu` | Rectified Linear Unit activation |
| **Dropout** | 0.2 | Applied to SimpleRNN layer |
| **Loss Function** | `binary_crossentropy` | Standard binary log-loss |
| **Optimizer** | `adam` | Adaptive Moment Estimation |
| **Batch Size** | 32 | Mini-batch size |
| **Max Epochs** | 10 | Early stopping configured with `patience=3` |
| **Validation Split** | 0.2 | 20,000 train samples / 5,000 validation samples |

### Training History & Convergence

The model was trained with `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`:

- **Epoch 1**: Loss: 2753023.75, Accuracy: 62.55% | Val Loss: 0.6167, Val Accuracy: 65.28%
- **Epoch 2**: Loss: 2938.30, Accuracy: 73.01% | Val Loss: 0.5931, Val Accuracy: 66.62%
- **Epoch 3**: Loss: 0.6457, Accuracy: 77.65% | Val Loss: 0.5757, Val Accuracy: 68.90%
- **Epoch 4**: Loss: 0.4222, Accuracy: 81.84% | Val Loss: 0.5208, Val Accuracy: 73.22%
- **Epoch 5**: Loss: 0.3363, Accuracy: 86.07% | Val Loss: 0.5194, Val Accuracy: 78.88%
- **Epoch 6**: Loss: 0.2571, Accuracy: 89.63% | **Val Loss: 0.4748**, **Val Accuracy: 79.28%** *(Best Model)*
- **Epoch 7**: Loss: 0.1959, Accuracy: 92.26% | Val Loss: 0.4959, Val Accuracy: 79.62%
- **Epoch 8**: Loss: 0.1567, Accuracy: 93.96% | Val Loss: 0.5578, Val Accuracy: 80.96%
- **Epoch 9**: Loss: 0.1273, Accuracy: 95.26% | Val Loss: 0.5587, Val Accuracy: 80.84%

> ℹ️ **Early Stopping Triggered**: Training halted after Epoch 9 because validation loss failed to improve beyond Epoch 6's benchmark (`0.4748`). The best weights from **Epoch 6** (~79.3% validation accuracy) were automatically restored.

---

## 🔄 How It Works (Pipeline)

When a review is entered in the web app, it passes through the following sequential pipeline:

```
[Raw Text Review]
       │
       ▼
1. Tokenization (`text_to_word_sequence`)
       │
       ▼
2. Word Index Mapping (IMDB Vocabulary Offset: word_id + 3; Unknowns -> 2)
       │
       ▼
3. Pre-Padding / Truncating (`pad_sequences(maxlen=500, padding='pre')`)
       │
       ▼
4. Dense Word Embeddings (`Embedding(10000, 128)`)
       │
       ▼
5. Sequence Processing (`SimpleRNN(128, activation='relu')`)
       │
       ▼
6. Binary Probability Output (`Dense(1, activation='sigmoid')`)
       │
       ▼
7. Thresholding & Confidence Calculation:
   - Score > 0.50 ──► "Positive" (Confidence = Score × 100%)
   - Score ≤ 0.50 ──► "Negative" (Confidence = (1 - Score) × 100%)
```

### IMDB Index Offsets
In Keras's IMDB dataset:
- `0`: Reserved for padding
- `1`: Reserved for sequence start
- `2`: Reserved for unknown / out-of-vocabulary (OOV) words
- `3`: Reserved / unused
- `4+`: Actual word indices (hence each word's index in the vocabulary dictionary is shifted by `+3`).

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hirvak/imdb-sentiment-analysis-simplernn.git
cd imdb-sentiment-analysis-simplernn
```

### 2. Set Up a Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage Instructions

### Running the Streamlit Web Application

To launch the web interface, execute:

```bash
streamlit run main.py
```

The app will start and provide a local URL (typically `http://localhost:8501`).

### Example Inputs & Outputs

| Sample Review | Predicted Sentiment | Prediction Score | Confidence |
| :--- | :---: | :---: | :---: |
| *"The movie was fantastic! I really enjoyed it."* | **Positive** | `0.8093` | `80.93%` |
| *"This movie was absolutely wonderful and charming! Highly recommend it."* | **Positive** | `0.7842` | `78.42%` |
| *"Worst movie ever made. Total waste of time and terrible acting."* | **Negative** | `0.1420` | `85.80%` |
| *"Boring storyline, poorly written script, and forgettable performances."* | **Negative** | `0.2105` | `78.95%` |

---

## 🔧 Troubleshooting & Known Issues

### 1. Keras / TensorFlow Deserialization Error (`time_major`)

**Problem:**
When loading the trained `.h5` model in modern versions of Keras or TensorFlow, you may encounter an error similar to:
```text
TypeError: SimpleRNN.__init__() got an unexpected keyword argument 'time_major'
```

**Cause:**
Models saved under certain TensorFlow/Keras versions store the parameter `'time_major': False` in the HDF5 layer configuration. Modern Keras 3+ has deprecated or removed this attribute from `SimpleRNN.__init__`.

**Solution:**
Run the included compatibility fixer:
```bash
python fix_model.py
```
This script reads `simple_rnn_imdb.h5`, strips all instances of `time_major` from the model's JSON metadata, and writes `simple_rnn_imdb_compatible.h5`. The web app (`main.py`) is already configured to load `simple_rnn_imdb_compatible.h5`.

---

### 2. Grappler / MLIR "Relu has an empty op name" Terminal Warnings

**Problem:**
During prediction, you may see terminal warning messages similar to:
```text
[MLIR/Grappler] Relu has an empty op name
I0000 00:00:... oneDNN custom operations are on.
```

**Cause & Harmlessness:**
- These messages are emitted by TensorFlow's Grappler optimizer and MLIR compiler pass while optimizing graph execution on CPU.
- **They are completely harmless**: they do not affect model accuracy, tensor calculations, or application functionality.

**How to suppress (optional):**
If you wish to keep your terminal clean, set the TensorFlow log level at the top of `main.py` before importing TensorFlow:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

---

## 🔮 Future Improvements

- [ ] **Advanced Recurrent Architectures**: Upgrade from `SimpleRNN` to **LSTM** (Long Short-Term Memory) or **GRU** (Gated Recurrent Unit) to better capture long-range dependencies and prevent vanishing gradients.
- [ ] **Bidirectional Layers**: Use `Bidirectional(LSTM(...))` to read review sequences in both forward and backward directions.
- [ ] **Transformer Models**: Implement state-of-the-art transformer architectures (such as BERT or DistilBERT) for superior semantic comprehension.
- [ ] **Cloud Deployment**: Deploy the Streamlit application to Streamlit Community Cloud, Hugging Face Spaces, or containerize via Docker.
- [ ] **Expanded Vocabulary**: Train on an expanded vocabulary (>20,000 words) or incorporate pre-trained word embeddings (GloVe / FastText).

---

## 📄 License & Author

### License
This project is open-source and licensed under the [MIT License](LICENSE).

### Author
- **Name**: Hirva Kansara
- **GitHub**: https://github.com/hirvak
---
