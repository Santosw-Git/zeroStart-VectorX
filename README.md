# ZeroStart-VectorX

ZeroStart-VectorX is a custom-built transformer-based language model inspired by GPT architectures. Unlike pre-existing GPT models, this model is built completely from scratch with unique modifications to improve text understanding and generation using vectorized embeddings.

## Overview

This project implements a language model with the following components:

- **Custom Transformer Blocks** – built from scratch to handle multi-head attention, feedforward layers, and normalization.
- **Vectorized Embeddings** – token and positional embeddings for efficient context representation.
- **Custom Training Pipeline** – PyTorch-based training using a custom dataloader for text data.
- **Next-Word Prediction** – generates text predictions based on given context.

The project demonstrates the full cycle of building a transformer: defining architecture, tokenization, batching, training, and generating text outputs.


### Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/Santosw-Git/zeroStart-VectorX.git
cd zeroStart-VectorX


### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training

```bash
python3 -m training.train
```

This will start the training process, print model statistics, and show intermediate outputs for text generation.

### 5. Training on Your Own Dataset

If you want to train ZeroStart-VectorX on your own dataset:

1. Add your dataset in `dataset/data.py`.
2. Ensure the data is properly formatted (e.g., text files or preprocessed token sequences).
3. Modify or create a new dataloader in `dataset/dataloader.py` to load your data.
4. Run training using the same command:

```bash
python3 -m training.train
```

This will use your custom dataset for training.

## Sample Output

Example of generated text after training:

```
I HAD always thought Jack Gisburn rather a peculiar fellow, but as I observed his movements...
```

## Contributing

I welcome contributions to this project! Here’s how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**

```bash
git checkout -b feature-name
```

3. **Make Your Changes**
4. **Commit and Push**

```bash
git commit -m "Description of changes"
git push origin feature-name
```

5. **Open a Pull Request** – Your contributions will be reviewed and merged after verification.

