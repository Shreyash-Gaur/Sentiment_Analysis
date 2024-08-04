---

# Social Media Sentiment Analysis

## Introduction
Sentiment analysis, a sub-field of Natural Language Processing (NLP), is one of the most popular topics in data science. This project focuses on analyzing social media sentiment to classify tweets as positive, negative, or neutral. The Sentiment140 dataset, which contains properly labeled and structured data, is used for this analysis. The Natural Language Toolkit (NLTK) is employed to perform the sentiment analysis.

## Approach
### Data Preprocessing
The Sentiment140 dataset contains 1.6 million tweets, each classified from 0 (negative) to 4 (positive). The initial dataset includes six fields: target, ids, date, flag, user, and text. For this project, only the label and tweet fields are retained. The dataset is then cleaned and preprocessed, including the removal of missing values.

### Word Embedding
To create word vectors, the GloVe (Global Vectors for Word Representation) algorithm is used. GloVe is an unsupervised learning algorithm that obtains vector representations for words based on global word-word co-occurrence statistics from a corpus.

### Model Selection
Three algorithms are chosen for classification:
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- Multinomial Naive Bayes

## Experiment Setup
### Data Analysis
Preprocessing includes data reduction, removal of stop words, and punctuation. The data is analyzed in terms of letter and word frequencies. Feature extraction methods such as bag-of-words and word embedding are applied.

### Model Training
The dataset is split into training (80%) and test (20%) sets. Six different models are tested:
- CNN Model-1: Conv1D = 64, Dense = 512, batch size = 1024
- CNN Model-2: Conv1D = 31, Dense = 256, batch size = 512
- LSTM Model-1: batch size = 1024
- LSTM Model-2: batch size = 512
- Multinomial Naive Bayes Model-1: Count Vectorizer
- Multinomial Naive Bayes Model-2: TF-IDF Vectorizer

### Evaluation Metrics
The models are evaluated using precision, recall, f1-score, AUC, and ROC.

## Experimental Results and Discussion
### Preprocessing Analysis
- Letter frequency analysis showed that the dataset does not follow the expected distribution of English text.
- Word frequency analysis identified the most common words in positive and negative tweets.

### Model Performance
- **LSTM Model-1** (batch size 1024) achieved the highest accuracy (78.9%).
- **CNN Model-1** (batch size 1024) was a close second with 78.1% accuracy.
- **Multinomial Naive Bayes Model-2** (TF-IDF Vectorizer) performed the worst with 75.8% accuracy.

### Training Times
- Naive Bayes models had the fastest training times.
- LSTM and CNN models with 512 batch size had better training times but slightly lower accuracy compared to those with 1024 batch size.

## Conclusion
The project successfully demonstrates the use of NLP techniques and machine learning models for social media sentiment analysis. The LSTM Model-1 with a batch size of 1024 is identified as the best-performing model in terms of accuracy. There is a trade-off between accuracy and training time, with models using a 512 batch size offering faster training times at a slight cost to accuracy.

## Figures
The report includes various figures and graphs to illustrate data distribution, letter and word frequencies, model performance, and ROC curves.

---