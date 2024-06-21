# Sentiment Analysis with Transformer Models

This project explores various Natural Language Processing (NLP) models to perform sentiment analysis on the Rotten Tomatoes movie reviews dataset. This final part of the individual assignment allowed me to experiment with a range of models provided by the Hugging Face Transformers library, including BERT, RoBERTa, and ultimately DeBERTa, each offering unique advantages and insights into the sentiment analysis task.

## Overview

Sentiment analysis is a critical component of understanding user-generated content, providing insights into the public perception of products, movies, and more. My goal was to leverage state-of-the-art NLP models to accurately classify movie reviews as positive or negative.

## Experimentation Journey

### BERT (Bidirectional Encoder Representations from Transformers)

- **Description**: I began the process with BERT, leveraging its ability to understand text from bidirectional context. BERT's pre-training on a vast corpus of text made it a strong contender for sentiment analysis.
- **Experimentation with Dense Layers**: I experimented with adding custom dense layers to BERT's architecture, aiming to enhance its feature extraction capabilities. This approach was a creative attempt to boost the model's performance by capturing more nuanced sentiment indicators.
- **Hyperparameter Tuning**: Delving into hyperparameter tuning, I explored various learning rates and batch sizes to optimize BERT's training process. This step was crucial for finding the sweet spot that maximized accuracy.

### RoBERTa (Robustly Optimized BERT Pretraining Approach)

- **Description**: Building upon BERT, RoBERTa fine-tunes the pre-training process, offering potentially improved performance for sentiment analysis tasks.
- **Experimentation with Dense Layers**: For RoBERTa, I further innovated by integrating additional dense layers, custom-tailored to refine sentiment analysis precision. This experimentation aimed at leveraging RoBERTa's robust training with enhanced model depth.
- **Hyperparameter Exploration**: I continued my journey of hyperparameter experimentation with RoBERTa, testing different configurations to unearth the optimal setup for our specific task. 

### DeBERTa (Decoding-enhanced BERT with Disentangled Attention)

- **Description**: DeBERTa introduces a novel approach to attention mechanisms and decoding, setting a new standard for understanding complex sentiments in text.
- **Experimentation with Dense Layers**: Implementing DeBERTa, I ambitiously added custom dense layers, pushing the model's capacity to interpret intricate sentiment nuances to new heights. This step was pivotal in achieving breakthrough performance.
- **Exploration with DeBERTa Large**: Additionally, I embarked on an exploration with DeBERTa Large, a more robust variant of the DeBERTa model, aiming to harness its extensive architecture for enhanced sentiment analysis capabilities, however due to my computing power I faced some issues with it.
- **Hyperparameter Tuning and AutoTransformers**: Not stopping at manual tweaks, I leveraged AutoTransformers for automated hyperparameter search, for optimal results however this is the part i struggled with and was not able to implement it. However by manual tweaks I was able to opyimise the overall performance.

## Conclusion and Model Selection

After extensive experimentation across various transformer models, I found the DeBERTa (Decoding-enhanced BERT with Disentangled Attention) model to significantly outperform its predecessors on the sentiment analysis task with the Rotten Tomatoes dataset. DeBERTa's innovative disentangled attention mechanism and enhanced decoding capabilities allow it to more effectively model the complex inter-word relationships present in natural language, providing a deeper understanding of context and sentiment. This breakthrough was evident in its  performance, marked by an impressive macro F1 score that solidified its place as my model of choice.

Moreover, the addition of custom dense layers to DeBERTa further amplified its feature extraction capabilities, enabling a nuanced capture of sentiment indicators that are crucial for accurately classifying movie reviews. This creative adjustment, combined with rigorous hyperparameter tuning and strategic experimentation, such as leveraging variopus libraries, propelled DeBERTa to its best accuracy and efficiency in sentiment analysis.


## Future Directions

- **Advanced Fine-tuning Techniques**: I am aware that a deeper delve into fine-tuning strategies usng autotransformers, especially focusing on the potential of layer-wise learning rate adjustments and differential learning rates will help me further refine DeBERTa's performance.
- **Dataset Expansion and Diversification**: Once again if we had the ability expanding the training corpus with more diverse datasets would allow us to build a model that not only excels in accuracy but also in understanding a wide range of sentiments across different contexts.

My journey through the landscape of transformer models has been one of relentless experimentation and creative problem-solving. From tweaking dense layers to harnessing AutoTransformers for hyperparameter optimization, each step has been a leap towards mastering sentiment analysis.