# phishing-email-detection-BERT
Detect phishing emails using a fine-tuned BERT model for binary text classification.

# 🛡️ Phishing Email Detection Using BERT

This project focuses on building a robust classification model capable of detecting phishing emails using the **BERT** language model. Leveraging transfer learning and the powerful language understanding capabilities of BERT, the goal is to differentiate between legitimate and phishing emails based solely on their textual content.

---

## 📌 Project Overview

- **Task**: Binary classification (Phishing vs. Non-Phishing Emails)
- **Model**: `bert-base-uncased` from Hugging Face Transformers
- **Dataset**: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) from Kaggle
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Environment**: Google Colab (GPU enabled)

---

## 📁 Dataset Details

- **Source**: CEAS_08.csv
- **Main Columns**:
  - `body`: Email content
  - `label`: Target class (`phishing` or `legitimate`)

---

## 🧪 Project Workflow

### 1. Dataset Preparation

- Load and inspect the dataset.
- Extract email text and corresponding labels.
- Split the data into training and validation sets (80/20 split).

### 2. Tokenization

- Use `BertTokenizer` to tokenize the email texts.
- Apply padding, truncation, and set a maximum length of 128 tokens.

### 3. Dataset & DataLoader Creation

- Wrap the tokenized inputs and labels into a PyTorch Dataset.
- Use DataLoader to create mini-batches for training.

### 4. Model Initialization

- Load a pre-trained BERT model (`BertForSequenceClassification`).
- Configure the model for binary classification.

### 5. Training Setup

- Optimizer: `AdamW`
- Scheduler: Linear scheduler with warmup
- Device: Automatically uses GPU if available

### 6. Model Training

- Train for multiple epochs (e.g., 5).
- Monitor training loss at each epoch.

### 7. Evaluation

- Predict labels on the validation set.
- Calculate and report the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

### 8. Inference on New Emails

A `predict_text()` function is implemented to classify new, unseen email texts and return whether the email is phishing or not.

---

## 🖥️ Running the Project

1. Clone this repository or open the notebook in Google Colab.
2. Enable GPU from:
   ```
   Runtime > Change runtime type > Hardware accelerator > GPU
   ```
3. Upload your `kaggle.json` and download the dataset using:
   ```python
   !pip install kaggle
   !kaggle datasets download -d naserabdullahalam/phishing-email-dataset
   !unzip phishing-email-dataset.zip
   ```
4. Follow the code cells in the notebook to preprocess the data, fine-tune the model, and evaluate results.

---

## 🔍 Example Usage

```python
text = "Please update your account credentials at the link below..."
label = predict_text(text)
print(f"This email is classified as: {label}")
```

---

## 📈 Results

Final model performance on the validation set (example):

- **Accuracy**: 94.5%
- **Precision**: 94.2%
- **Recall**: 94.7%
- **F1-Score**: 94.4%

> *Note: Actual performance may vary depending on the data split and training parameters.*

---

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Kaggle Phishing Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

---

## 👨‍💻 Author

This project was developed as part of a hands-on exercise to explore deep learning applications in **cybersecurity and natural language processing (NLP)**.

---
