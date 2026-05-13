# Google Maps Review Rating Prediction

> Comparative NLP and Machine Learning study for predicting Google Maps review ratings using classical ML, BiLSTMs, and transformer-based architectures.

---

## Project Overview

This project focuses on predicting Google Maps review ratings (1–5 stars) from raw review text using Natural Language Processing and Machine Learning techniques. The work explores the progression from traditional vector-space models to sequence-based deep learning and transformer architectures.

The project compares:

* TF-IDF + classical machine learning models
* Sequence-based deep learning models (BiLSTM)
* Transformer-based architectures (DistilBERT and RoBERTa)
* Topic modelling using Latent Dirichlet Allocation (LDA)

The objective was not only to maximize classification performance, but also to analyze how different modelling approaches capture sentiment, context, and semantic meaning within review text.

---

## Key Results

| Model               | Accuracy   | Macro F1 Score |
| ------------------- | ---------- | -------------- |
| Logistic Regression | 0.6568     | 0.5108         |
| SVM                 | 0.6503     | 0.4910         |
| BiLSTM              | 0.6597     | 0.5359         |
| DistilBERT          | 0.6567     | 0.6031         |
| RoBERTa             | **0.6703** | **0.6159**     |

### Highlights

* RoBERTa achieved the best overall performance.
* Transformer models significantly outperformed classical TF-IDF-based approaches.
* BiLSTM improved contextual understanding over Bag-of-Words models.
* Bigram TF-IDF features consistently improved traditional ML performance.
* Macro F1-score was prioritized due to heavy class imbalance.

---

## Dataset

The dataset contains **288,000 Google Maps reviews** with associated star ratings.

### Dataset Characteristics

* No missing values
* Significant class imbalance
* Majority class: 1-star reviews
* Minority class: 5-star reviews

### Example Reviews

| Review                                 | Rating |
| -------------------------------------- | ------ |
| "Amazing service and friendly staff."  | 5      |
| "Worst experience ever, staff made us wait hours. AC was barely working, bad ambience" | 1      |

### Key Observations

* 1-star reviews were significantly longer and more descriptive.
* Negative reviews frequently referenced delays, appointments, and billing issues.
* Positive reviews emphasized friendliness, professionalism, and helpful staff.

---

## Exploratory Data Analysis

The project included extensive exploratory analysis and visualization.

### Review Length Analysis

* 1-star reviews had a median length of ~51 words.
* 5-star reviews were typically shorter and more concise.
* Longer negative reviews often described detailed failure narratives.

### Frequent Keywords

#### 1-Star Reviews

```text
never, rude, worst, wait, hours, appointment, pay
```

#### 5-Star Reviews

```text
great, amazing, friendly, helpful, professional
```

### Important Insight

The overlap of words such as "help" across positive and negative reviews exposed a major limitation of Bag-of-Words models:

```text
"helpful staff"
vs
"no help at all"
```

This motivated the use of sequence models and transformers that preserve contextual relationships.

---

## Text Preprocessing

Different preprocessing pipelines were used depending on the model architecture.

### Classical ML Pipeline

Applied preprocessing:

* Lowercasing
* Tokenization
* Stop-word removal
* Lemmatization
* Stemming
* Punctuation removal
* TF-IDF vectorization

### Deep Learning / Transformer Pipeline

Minimal preprocessing was intentionally used:

* Lowercasing
* Whitespace cleanup

Aggressive preprocessing was avoided because transformers rely heavily on contextual information and word order.

---

## Feature Engineering

Several text representation techniques were explored.

### Vector Representations

* Binary Representation
* Bag of Words (BoW)
* TF-IDF
* N-grams
* Word Embeddings (Word2Vec)

### N-Gram Statistics

| Feature Type | Number of Features |
| ------------ | ------------------ |
| Unigrams     | 138,760            |
| Bigrams      | 3,027,444          |
| Trigrams     | 9,699,147          |

### Why TF-IDF?

TF-IDF provided the best balance between:

* Computational efficiency
* Feature importance weighting
* Classification performance
* Sparse representation handling

---

## Classical Machine Learning Models

The following models were trained using TF-IDF representations:

* Logistic Regression
* Linear SVM
* Naive Bayes
* SGD Classifier
* Ridge Classifier

### Hyperparameter Tuning

GridSearchCV with 3-fold cross-validation was used for tuning:

* TF-IDF max features
* N-gram ranges
* Regularization strength
* Learning rates
* Smoothing parameters

### Best Classical Model

```text
Logistic Regression
Accuracy: 0.6568
Macro F1: 0.5108
```

Linear models performed well due to the high-dimensional sparse TF-IDF feature space.

---

## BiLSTM Model

A Bidirectional LSTM was implemented to capture sequential dependencies.

### Architecture

```text
Embedding Layer
        ↓
Bidirectional LSTM
        ↓
Dropout
        ↓
Dense Layer (ReLU)
        ↓
Softmax Output
```

### Best Configuration

| Parameter           | Value  |
| ------------------- | ------ |
| Vocabulary Size     | 10,000 |
| Sequence Length     | 160    |
| Embedding Dimension | 200    |
| LSTM Units          | 128    |
| Batch Size          | 64     |
| Epochs              | 5      |

### Performance

```text
Accuracy: 0.6597
Macro F1: 0.5359
```

The BiLSTM outperformed all traditional ML models by leveraging contextual sequence information.

---

## Transformer Models

Transformer architectures were fine-tuned using pretrained contextual embeddings.

### Models Used

* DistilBERT
* RoBERTa

### DistilBERT Results

| Max Length | Learning Rate | Epochs | Accuracy | F1     |
| ---------- | ------------- | ------ | -------- | ------ |
| 160        | 2e-5          | 2      | 0.6567   | 0.6031 |

### RoBERTa Results

| Max Length | Learning Rate | Epochs | Accuracy   | F1         |
| ---------- | ------------- | ------ | ---------- | ---------- |
| 160        | 2e-5          | 2      | **0.6703** | **0.6159** |

### Key Insight

Transformer models substantially improved contextual understanding by using:

* Self-attention mechanisms
* Pretrained contextual embeddings
* Long-range dependency modelling

RoBERTa achieved the strongest performance across all experiments.

---

## Topic Modelling (LDA)

Latent Dirichlet Allocation (LDA) was applied to analyze thematic differences between 1-star and 5-star reviews.

### 1-Star Reviews

Main themes:

* Billing disputes
* Long waiting times
* Staff rudeness
* Medical delays
* Automotive repair complaints

### 5-Star Reviews

Main themes:

* Professionalism
* Friendly staff
* Helpful service
* Emotional satisfaction
* Positive interpersonal experiences

### Important Finding

Negative reviews focused on:

```text
Objective process failures
```

Positive reviews focused on:

```text
Subjective customer experience
```

---

## Kaggle Performance

| Model               | Kaggle Score |
| ------------------- | ------------ |
| Logistic Regression | 61.7         |
| BiLSTM              | 62.7         |
| DistilBERT          | 62.9         |
| RoBERTa             | **66.6**     |

The Kaggle leaderboard results closely matched cross-validation findings.

---

## Tech Stack

### Languages & Libraries

* Python
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* Pandas
* NumPy
* NLTK
* Matplotlib
* Seaborn
* Gensim

### Models & Architectures

* TF-IDF
* Logistic Regression
* Linear SVM
* BiLSTM
* DistilBERT
* RoBERTa
* LDA Topic Modelling

---

## Conclusion

This project demonstrates the progression from traditional NLP pipelines to modern transformer-based approaches for sentiment and rating prediction.

While TF-IDF-based linear models provided strong baselines with low computational cost, transformer architectures achieved significantly better performance by capturing semantic relationships and contextual meaning within review text.

The results highlight the importance of context-aware representations in modern NLP classification tasks.

---

## Contributors

| Name                      | Contribution                                   |
| ------------------------- | ---------------------------------------------- |
| Mohamed Hafeez            | EDA and Topic Modelling                        |
| Mohamed Ihsan             | Text Processing and Feature Engineering        |
| Sri Sai Vaishnavi Chintha | Predictive Modelling and Hyperparameter Tuning |

---

## References

1. Liu, Z. (2020). Yelp Review Rating Prediction: Machine Learning and Deep Learning Models.
2. Puh, K. & Bagić Babac, M. (2023). Predicting sentiment and rating of tourist reviews using machine learning.
3. Kusal, S. et al. (2023). Sentiment analysis of product reviews using deep learning and transformer models.
4. Areshey, A. & Mathkour, H. (2024). Exploring transformer models for sentiment classification.

---

## Report

[View Report](F20AA_CW2.pdf)
