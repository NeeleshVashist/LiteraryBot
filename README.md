# LiteraryBot

![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![License](https://img.shields.io/github/license/NeeleshVashist/LiteraryBot)

LiteraryBot is a fun and interactive Generative AI Language Model project that brings literary text generation to life. Leveraging the power of LSTM (Long Short-Term Memory) neural networks, GPT-2 Medium Model and the rich text corpus from Project Gutenberg, LiteraryBot can generate coherent and contextually relevant text based on a given prompt.

## Course Information

This project is part of the "Software Tools and Emerging Technologies for AI and ML" class (AML 3304).

## Team PyCoders - Team Members:

- Neelesh Vashist - C0858518
- Rohit Kumar - C0859060
- Mukul Bisht - C0857928
- Saurabh Singh - C0859334

## Project Introduction

In this project, we have created a generative chatbot using LSTM (Long Short-Term Memory) neural networks, GPT-2 Medium Model and leveraged the rich text corpus from Gutenberg Books. The goal of this project is to develop a language model that can generate coherent and contextually relevant text based on a given prompt. We aim to explore the capabilities of both an LSTM-based model and a pre-trained GPT-2 (Generative Pre-trained Transformer 2) model.

## Project Overview

The objective of this project was to build a Minimal Viable Product (MVP) of a Generative AI Language Model. The team chose to use Python and popular libraries like TensorFlow and Keras for model development. The main tasks involved data collection, preprocessing, model building, and evaluation.

## Installation and Usage

1. **Clone the Repository**: Dive into the world of LiteraryBot by cloning this repository:
   ```bash
   git clone https://github.com/NeeleshVashist/LiteraryBot.git
   cd LiteraryBot
   ```
   
2. **Run the Jupyter Notebook**: Enter the magical realm of text generation by running the Jupyter Notebook:
   ```bash
   jupyter notebook LiteraryBot.ipynb
   ```

3. Train the Models: Within the notebook, embark on a literary journey as you train the LSTM-based model and fine-tune the illustrious GPT-2 model.

## Dataset

For training the language model, we utilized the Gutenberg corpus, a vast collection of literary works from Project Gutenberg. The dataset was preprocessed to remove unwanted characters, convert text to lowercase, and tokenize the text.

## Model Architectures

Two models were explored in this project: 
- LSTM-based Language Model: A simple neural network with a single LSTM layer.
- GPT-2 Language Model: A pre-trained generative language model fine-tuned on the Gutenberg corpus.

The LSTM-based model was built from scratch, whereas the GPT-2 model was imported using the Hugging Face library.

## Data Preprocessing

The text data from the Gutenberg corpus underwent several preprocessing steps. We used the NLTK library for tokenization, removed stop words, and performed basic text cleaning.

## Training

Both models were trained on a GPU-enabled machine. The LSTM model was trained for 50 epochs, and the GPT-2 model was fine-tuned for 5 epochs.

## Results and Evaluation

### Training Loss and Accuracy
The training process was conducted over 50 epochs. The model's training loss and accuracy were recorded after each epoch. The training loss started at 1.9947 and gradually decreased over the epochs, reaching a final value of 1.4128. Similarly, the training accuracy began at 0.4113 and improved with each epoch, reaching a final value of 0.5639.

### Validation Loss and Accuracy
During training, the model's performance on the validation set was also recorded after each epoch. The validation loss started at 1.7354 and decreased steadily, reaching a final value of 1.4150. The validation accuracy started at 0.4840 and improved throughout training, reaching a final value of 0.5668.

### Overfitting and Underfitting Analysis
Overfitting occurs when a model becomes too specialized in learning from the training data, resulting in poor generalization to unseen data. Underfitting, on the other hand, happens when the model fails to learn the underlying patterns in the data, leading to suboptimal performance even on the training set.

Based on the recorded training and validation loss and accuracy values, we can make the following observations:

- The training loss decreases consistently throughout the training process, indicating that the model is learning from the training data effectively.
- The validation loss also decreases, but there is a slight increase in validation loss in the later epochs (from epoch 40 to 50), indicating a possible sign of overfitting.
- The training accuracy improves over the epochs, showing that the model is capturing the patterns in the training data.
- The validation accuracy also improves, but there is a slight fluctuation in the later epochs, suggesting potential overfitting.

## Conclusion

The model has shown promising performance in terms of training accuracy and validation accuracy, achieving approximately 56.39% and 56.68%, respectively, at the end of training. However, there are signs of overfitting, especially during the later epochs, where the validation loss and accuracy start to fluctuate.

To address overfitting, several techniques can be employed, such as regularization, dropout layers, and early stopping. Additionally, fine-tuning hyperparameters or adjusting the model architecture may also help improve generalization performance.

Further analysis and experimentation are recommended to optimize the model's performance and mitigate overfitting issues.

## Collaboration and Project Management

The team used Trello for project management, creating a shared board with task lists for each team member. GitHub was employed for version control, allowing seamless collaboration.

## Lessons Learned

Throughout the project, the team learned the importance of data preprocessing and the impact of model architecture on the generated text. The benefits of leveraging pre-trained models were evident in the case of GPT-2, which achieved better text generation results with minimal fine-tuning.

## Future Enhancements

To improve the language model's performance, future enhancements could involve exploring larger datasets, experimenting with different model architectures, and fine-tuning GPT-2 on domain-specific corpora.

## References

- Hugging Face Transformers Library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- Hugging Face GPT2 Medium Model: [https://huggingface.co/gpt2-medium](https://huggingface.co/gpt2-medium)
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- NLTK Documentation: [https://www.nltk.org/](https://www.nltk.org/)
- Project Gutenberg: [https://www.gutenberg.org/](https://www.gutenberg.org/)
- ChatGPT: [https://chat.openai.com/](https://chat.openai.com/)

## License

LiteraryBot is licensed under the [MIT License](LICENSE).
