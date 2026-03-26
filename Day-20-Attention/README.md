## Day 20 - Bi-LSTM and Attention Mechanism

Today I focused on predicting the **nationality** of a person by their **surname**.

I used the **surname-nationality dataset** from Kaggle.

I tried doing this with 3 different Architectures,
* Unidirectional LSTM
* Bidirectional LSTM
* Attention-based Bidirectional LSTM

Observed only slight improvement in each architecture over the previous.

Also observed a significant difference in Accuracy when looking at the top prediction, compared with the Accuracy when looking at the top 3 predictions, in each architecture.

### Basic LSTM

```
Epoch 1/50, Training Loss: 2.6460
...
Epoch 50/50, Training Loss: 0.8298

Evaluating Model
Accuracy on Testing data: 54.85%
```

```
Top-3 Accuracy: 72.95%
```

### Bidirectional LSTM

```
Epoch 1/50, Training Loss: 2.1456
...
Epoch 50/50, Training Loss: 0.7306

Evaluating Model
Accuracy on Testing data: 55.64%
```

```
Top-3 Accuracy: 73.51%
```

### Attention-based LSTM

```
Epoch 1/50, Training Loss: 2.0913
...
Epoch 50/50, Training Loss: 0.7045

Evaluating Model
Accuracy on Testing data: 56.59%
```

```
Top-3 Accuracy: 74.27%
```