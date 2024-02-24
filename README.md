
# README: Text Classification Project

## Overview

This project involved developing a classical machine learning text classification model that would be capable of accurately categorizing text data. The primary objective was to experiment with various natural language processing (NLP) techniques learned in class and explore machine learning models to find the most effective combinations. We were challenged with finding the higest F1 macro score for the models while making sure they did not overfit the data. The overall process involved setting up the environment, cleaning and preparing the dataset, experimenting with different preprocessing techniques and models, and finally  evaluating their performance.

## Setup and Initial Steps

### Installing Libraries

I begain the project by installing essential Python libraries for data manipulation, NLP, and machine learning, including:

- `datasets` for accessing our primary text datasets.
- `nltk` and `textblob` for natural language processing.
- `scikit-learn` for machine learning models and pipeline creation.
- `pandas` and `numpy` for data manipulation.
- `matplotlib` and `seaborn` for visualization.
- Additional utilities like `BeautifulSoup` for HTML tag removal in preprocessing.

### Data Preparation

#### Datasets

I decided to use the 'imdb' dataset as an additional dataset along with the required dataset which was 'roten tomatoes' as there would be much more data for training the models and hence help the model learn better and provide a better score.

1. **Rotten Tomatoes Movie Reviews:** For training and evaluating our models.
2. **IMDb Movie Reviews:** To augment our training data.

The datasets were combined to form a larger, more diverse training set. I made sure to removed duplicates to ensure the model's generalizability and fairness in evaluation.

#### Data Splitting

The combined dataset was split into training and testing sets, with the test set exclusively derived from the Rotten Tomatoes dataset to assess the model's performance on unseen data reliably.

## Preprocessing and Feature Extraction

### Linguistic Processing

Several linguistic processors were experimented with, focusing on tasks such as:

- HTML tag removal using `BeautifulSoup`.
- Tokenization, lemmatization, and stop words removal with `nltk`.
- Sentiment feature extraction using `TextBlob`.
- Vector representations with '`Word2Vec`.

The choice of linguistic processors was driven by the need to clean the text data effectively and to explore the impact of sentiment analysis on classification performance. Although I tried several linguistic processors they did not seem to make as much of a difference on the final score, in fact trying to use 'TextBlob' for feature extraction and 'Word2vec' for vector represntation of similar word relationships, made my score way lower which is why I decided to leave them out in the final model.

### Vectorization Techniques

Different vectorization methods were explored to convert text into numerical features suitable for machine learning models:

- **TF-IDF Vectorizer:** To capture term importance.
- **Count Vectorizer:** For raw word counts.
- Attempts were made to use a Voting Classifier to combine models effectively.

I found that TF_IDF models performed the best with the all the models that I chose to implement.

## Model Exploration

Several machine learning models were evaluated, including:

- **SGDClassifier (SDGC):** For its versatility and efficiency in handling large datasets.
- **Naive Bayes:** Due to its simplicity and effectiveness in text classification tasks.
- **Random Forest:** For its robustness and ability to capture non-linear relationships.

To prepare the data for modeling, I employed a LinguisticPreprocessor followed by TfidfVectorizer in the Pipeline. This preprocessing step was crucial for normalizing the text data, including tasks such as removing HTML tags, punctuations, and stop words, in addition to lemmatization. This ensured that my model learned from clean, meaningful textual features. Subsequently, the TfidfVectorizer transformed the preprocessed text into numerical vectors, emphasizing words that are important to the context of a document while mitigating the influence of frequent but less informative words.

I expereminted with several models including the ones above and found that the SDGCClassifier performed best and was able to accurately classify the textual data into their respective labels. I also ran GridSearch and tried to get the best hyperparameters but since this gave me the same score as simply applying the pipeline I chose to leave it out due to the amount of time it took to run and search for the hyperparameters. 

While developing the text classification model, I also tried an innovative approach by integrating a stacked ensemble approach. I was trying to leverage the complementary strengths of different machine learning algorithms to enhance our model's predictive performance. The stacked model comprised of two base learners, SGDClassifier and MultinomialNB, chosen for their proven efficacy in handling text data and distinct learning mechanisms. This time the data was first preprocessed and vectorized then the base learners were trained on this prepared dataset, with their predictions serving as input features for a LogisticRegression meta-learner. This stacking strategy was chosen to capture different aspects of the data, potentially leading to improved performance. However this time my score dropped by 0.02 so I decided to stick to the initial pipeline. 

## Evaluation

The models' performance was rigorously evaluated using several metrics:

- **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.
- **Classification Report:** Including precision, recall, and F1-score for a detailed performance breakdown.
- **Accuracy Score:** To provide a quick, overall performance metric.

Since the final score that would be the judge for the assignment was the Macro F1 score I tried my best to use as many possible diffreent strategies to improve this while also making sure to not overfit the data and provide reliable results. 

## Conclusion and Model Selection

After thorough experimentation and evaluation, the best-performing model was chosen based on a balance of accuracy, F1-score, and generalization capability to unseen data. The choice was influenced by the model's ability to handle the nuances of natural language effectively, its computational efficiency, and its interpretability.

The process of data preparation, linguistic processing, model exploration, and rigorous evaluation taught me how complex the nature of designing a NLP system truely is. The iterative nature and systematic approach for preprocessing, model selection and performance evaluation were a bit tedious but allowed me to learn from scratch and understand the reasoning behind why each step performed the way it did.