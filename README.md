# Next-Word Prediction using MLP (Word-Level Text Generator)

**[GitHub Repository](https://github.com/HARSHA219219/Next-Word-Prediction-using-MLP-Word-Level-Text-Generator-.git)**

---

## Project Overview 

The project builds a word-level text generation system that predicts the next word in a sequence using a Multi-Layer Perceptron (MLP). Instead of using sequence models like RNNs or Transformers, the system learns word embeddings and uses an MLP to model relationships between words based on context windows.

The model is trained on both natural language and structured/domain text, allowing comparison of how different language types influence learning, predictability, and generated output quality. The project also includes visualization of learned embeddings and an interactive Streamlit interface for real-time text generation.

---

## Objectives

- To design a word-level next-word prediction pipeline using MLP-based language modeling.
- To construct and analyze vocabulary, context windows, and embedding representations.
- To compare model behavior on natural vs structured datasets and study predictability differences.
- To visualize embedding clusters using t-SNE for semantic interpretation.
- To develop an interactive Streamlit interface enabling configurable generation and temperature-based sampling.

---

## Datasets Used
1. Category I — Natural Language (e.g., Sherlock Holmes / Paul Graham / Wikipedia)
2. Category II — Structured / Domain Text (e.g., Python code / Math text / docs etc.)

---

## Data and Model Preparation

- The `datasets/` directory contains the natural-language and structured text corpora used for training and evaluation.
- All preprocessing steps — including tokenization, vocabulary construction, and context-window generation — are implemented and explained in the accompanying Jupyter notebooks.
- Word embeddings and trained MLP model checkpoints are precomputed and stored to enable fast loading and inference during experimentation and deployment.

## Project Structure

```
Next-Word-Prediction-MLP/
│
├── Dataset/                         # Raw and processed training corpora
├── Images of App Created/           # Screenshots of Streamlit app
├── Model_Training/                  # Notebooks / scripts used for training
├── Next-Word Prediction using MLP (Word-Level ...)/   # Report / documentation
├── app.py                           # Streamlit application for text generation
├── requirements.txt                 # Python dependencies
│
├── holmes_mlp64_relu_config.json    # Config for MLP (Holmes dataset, ReLU)
├── holmes_mlp64_relu_state.pt       # Trained model weights (ReLU)
├── holmes_mlp64_relu_vocab.json     # Vocabulary file (ReLU model)
│
├── holmes_mlp64_tanh_config.json    # Config for MLP (Holmes dataset, Tanh)
├── holmes_mlp64_tanh_state.pt       # Trained model weights (Tanh)
├── holmes_mlp64_tanh_vocab.json     # Vocabulary file (Tanh model)
│
├── linux_mlp64_relu_config.json     # Config (Linux dataset, ReLU)
├── linux_mlp64_relu_state.pt        # Trained model weights (ReLU)
├── linux_mlp64_relu_vocab.json      # Vocabulary file (ReLU model)
│
├── linux_mlp64_tanh_config.json     # Config (Linux dataset, Tanh)
├── linux_mlp64_tanh_state.pt        # Trained model weights (Tanh)
└── linux_mlp64_tanh_vocab.json      # Vocabulary file (Tanh model)
```
## How to use
Install dependencies:
```bash
pip install -r requirements.txt
```
Launch Streamlit app
```bash
streamlit run app.py
```

---

## Results and Observation 

The detailed results and observations are documented in the project report located in the ```Next-Word Prediction using MLP (Word-Level Text Generator)/Next-Word Prediction using MLP (Word-Level Text Generator).pdf``` directory.

## Streamlit Application

- Provides an interactive interface for experimenting with the trained models.
- Supports control over:
  - context length
  - embedding dimension
  - activation function
  - model selection
  - temperature-based sampling
- Allows users to observe how different configuration choices affect generated text.
- The Streamlit interface and source code are implemented in app.py and are located in the root directory of the project.
- Screenshots of the interface are included below for reference.

![image](https://github.com/HARSHA219219/Next-Word-Prediction-using-MLP-Word-Level-Text-Generator-/blob/main/Images%20of%20App%20Created/Screenshot%206.png)

## 👤 Author

**Vadithya Harsha Vardhan Nayak**\
Developer & Machine Learning Enthusiast

🔗 LinkedIn:[Vadithya Harsha Vardhan Nayak](linkedin.com/in/vadithya-harsha-vardhan-nayak-a37324372) \
linkedin.com/in/vadithya-harsha-vardhan-nayak-a37324372

📌 If you found this project useful or have suggestions, feel free to connect with me on LinkedIn!

---
