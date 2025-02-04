
# 📢 Sentiment Analysis with LSTM Neural Networks

## 📝 **Project Overview**

This project focuses on performing **Sentiment Analysis** using **Natural Language Processing (NLP)** techniques and deep learning models. We employ **Long Short-Term Memory (LSTM)** neural networks to classify text data into **positive**, **neutral**, and **negative** sentiments. The goal is to analyze the underlying emotion in text data for applications like social media monitoring, customer feedback analysis, and more.

---

## 🎯 **Objectives**

- Clean and preprocess text data for NLP tasks.
- Apply **tokenization** and **padding** techniques to prepare data for deep learning models.
- Build and train an **LSTM-based Neural Network** using TensorFlow and Keras.
- Improve model performance with **hyperparameter tuning** using GridSearchCV.
- Predict sentiments on new, unseen text data.

---

## 📂 **Dataset Description**

- **Dataset:** Sentiment-labeled text dataset (`Sentiment.csv`)
- **Features:**
  - `text`: Textual data containing sentences.
  - `sentiment`: Labels indicating the sentiment (*Positive*, *Neutral*, *Negative*).
- **Target Variable:** `sentiment`

---

## ⚙️ **Technologies Used**

- **Python Libraries:**
  - `pandas`, `numpy` for data manipulation
  - `re` for regular expressions (text preprocessing)
  - `scikit-learn` for data splitting, preprocessing, and model evaluation
  - `TensorFlow` and `Keras` for building LSTM models
  - `SciKeras` for hyperparameter tuning with GridSearchCV

- **Deep Learning Techniques:**
  - **Embedding Layer**: To convert words into dense vectors.
  - **LSTM Layer**: To capture long-term dependencies in text sequences.
  - **Dense Layer with Softmax**: For multi-class classification.

---

## 🧠 **Model Architecture**

- **Embedding Layer:** Input dimension = 2000, Output dimension = 128
- **LSTM Layer:** 196 neurons with 20% dropout
- **Dense Layer:** 3 output neurons (*Positive*, *Neutral*, *Negative*) using **Softmax** activation
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## 📊 **Project Workflow**

1. **Data Preprocessing:**
   - Convert text to lowercase
   - Remove special characters using regular expressions
   - Remove retweets and unwanted text

2. **Tokenization & Padding:**
   - Tokenize the text data (max 2000 words)
   - Apply padding to ensure uniform input length

3. **Model Training:**
   - Build and compile an LSTM-based Sequential model
   - Train the model on the training set (67%) and evaluate on the test set (33%)

4. **Hyperparameter Tuning:**
   - Apply **GridSearchCV** to optimize batch size and number of epochs

5. **Prediction:**
   - Predict sentiments on new, unseen text data

---

## 🚀 **How to Run the Project**

1. **Clone the Repository:**
   ```bash
   git clone [GitHub Repository Link]
   ```

2. **Install Dependencies:**
   ```bash
   pip install tensorflow scikit-learn scikeras pandas numpy matplotlib
   ```

3. **Run the Notebook or Script:**
   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

4. **Predict Sentiment on New Data:**
   ```python
   sentence = ['This product is amazing!']
   sentiment_probs = model.predict(pad_sequences(tokenizer.texts_to_sequences(sentence)))
   print(sentiment_probs)
   ```

---

## 📈 **Model Performance**

- **Initial Model Accuracy:** ~67%
- **After Hyperparameter Tuning (GridSearchCV):** Improved accuracy to ~50% (depending on batch size and epochs)
- **Best Parameters:** `{ 'batch_size': 20, 'epochs': 1 }`

---

## 🔍 **Future Improvements**

- Experiment with advanced models like **Bidirectional LSTM** or **GRU**.
- Incorporate **pre-trained word embeddings** (e.g., Word2Vec, GloVe) for improved performance.
- Add **attention mechanisms** to capture important words in long texts.
- Perform **cross-validation** for more robust evaluation.

---

## 🙌 **Contributions**

Contributions are welcome! Feel free to fork the repository, improve the model, and submit pull requests.

---

**Author:** Nikitha Kadaparthi  
**GitHub:** [GitHub Profile](https://github.com/Nikithakadaparthi)  
**LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/nikitha-kadaparthi-4a42321a8/)

> *"Empowering machines to understand human emotions through sentiment analysis."*

