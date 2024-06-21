# README: Sentiment Analysis on Rotten Tomatoes Movie Reviews

## Introduction and background 

The second part of this assignment allowed me to delve deeper into natural lanaguage processing with the same aim to to perforrm a sentiment analysis on the Rotten tomatoes dataset. The primary goal of this project was to develop a model capable of distinguishing between positive and negative sentiments expressed in movie reviews. Throughout this process, I explored a variety of neural network architectures and techniques, each offering unique insights and learning opportunities.
Given the subjective nature of movie reviews, this task involved the computational study of people's opinions, sentiments, emotions, and attitudes within the text and challenged me to find the most accurate model in order to classify them.

## Approach and Experiments

Initially, I experimented with several neural network architectures to understand their strengths and limitations in the context of text classification. Here's a brief overview of the techniques I tried:

1. **LSTM (Long Short-Term Memory) Networks**: Known for their effectiveness in capturing long-range dependencies in sequential data, LSTM networks were my first choice. While they proved well in handling sequences, the complexity and computational ineeficiency was a bit challenging.

2. **GRU (Gated Recurrent Unit) Networks**: As a simplified variant of LSTM, GRUs offered a more streamlined architecture, which facilitated faster training. However, the performance trade-off became evident when dealing with this dataset. 

3. **CNN (Convolutional Neural Network) with Static Embeddings**: Transitioning to CNNs, I leveraged their capability to capture local feature correlations through convolutional filters. By integrating pre-trained static embeddings, the network was able to train well and give me the best score out of all the models that I tried. 

4. **Ensemble Models**: I also dabbled with ensemble models by trying to leverage the power of both CNN and LSTM together however for the amount of effort and time it took to train the results were not better than the other models. 

## Final Model Selection

After a series of experiments and evaluations, I converged on a CNN model with static embeddings as the final choice. This decision was driven by several key factors:

- **Effectiveness in Text Classification**: The CNN demonstrated a robust ability to extract meaningful patterns and features from the textual data, proving to be highly effective in classifying sentiments.

- **Computational Efficiency**: Compared to LSTM and ensemble models, the CNN offered a more computationally efficient architecture, enabling faster training without compromising on performance.

- **Interpretability**: The convolutional layers in the CNN provided a degree of interpretability, allowing me to understand which features were most influential in determining sentiments.

## Implementation and Functionality

Hyperparameter Tuning: I delved into hyperparameter tuning to optimize the CNN model's performance, methodically adjusting parameters like the number of convolutional filters, kernel size, and learning rate. This meticulous process was instrumental in enhancing the model's accuracy.

TextDataset Class: This class is important for preprocessing, wrapping the Rotten Tomatoes dataset into a PyTorch-friendly format. It tokenizes the text and converts it into numerical tensors, setting the stage for model training.

TextCNN Class: The TextCNN class defines the architecture comprising an embedding layer, convolutional layers, ReLU activations, pooling, and a dense layer for classification. This architecture allowed me to distinguish sentiments within the reviews.

prepare_data Function: An essential part of the code that helps with data loading and preprocessing, facilitating an efficient pipeline from raw text to tokenized tensors ready for the model.

train_and_validate Function: This function demonstrates the training logic, iteratively optimizing the model while monitoring its performance. Incorporating validation within training allowed me to assess the model's generalization and make informed decisions on early stopping or further tuning.

evaluate_test_set Function: Beyond training, this function rigorously evaluates the model on unseen test data, providing a Macro F1 score that was required to judge the final model's effectiveness in real-world sentiment analysis.

main Function: This essentiall orchestrates the code by tying it all together, ensuring a seamless execution flow from data preparation to model evaluation.

## Conclusion

The journey to develop this sentiment analysis model was both challenging and rewarding. Through the process of experimenting with various neural network architectures, I gained valuable insights into their respective advantages and trade-offs. The chosen CNN model, with its balance of performance, efficiency, and interpretability, proved to be an effective tool for classifying sentiments in movie reviews. This project not only enhanced my practical skills in machine learning and natural language processing but also deepened my appreciation for the nuanced task of sentiment analysis.