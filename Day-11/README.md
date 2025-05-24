## Day 11

### Handling Imbalanced Data

Implemented a `LogisticRegression` model to detect if a credit-card transaction is fraud or not fraud. Picked this dataset to practice handling data imbalance.

First tried it without any resampling.

The Output was:

```
Without Resampling the Data ... 
Creating our Model ... 
Fitting the model ... 
Model training completed.
Confusion Matrix:
[[113717     15]
 [    75    116]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113732
           1       0.89      0.61      0.72       191

    accuracy                           1.00    113923
   macro avg       0.94      0.80      0.86    113923
weighted avg       1.00      1.00      1.00    113923

```

Without imbalance handling there were some `False Positives` i.e, samples that were fraud, but detected as not fraud.

#### **Random Over Sampler**

Then, I tried using `RandomOverSampler` to balance the classes in the dataset (it balances by duplicating the samples of minority class).

The Output was:
```
With Random Over Sampling ... 
Creating our Model ... 
Fitting the model ... 
Model training completed.
Confusion Matrix:
[[111043   2689]
 [    17    174]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99    113732
           1       0.06      0.91      0.11       191

    accuracy                           0.98    113923
   macro avg       0.53      0.94      0.55    113923
weighted avg       1.00      0.98      0.99    113923

```

#### **SMOTE**

Then, I tried it with `SMOTE` (Synthetic Minority Over-sampling Technique), it is basically `Data-Augmentation` for the minority class.

The Output was:
```
With SMOTE
Creating our Model ... 
Training our model ... 
Training completed
Confusion Matrix:
[[110826   2906]
 [    16    175]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.97      0.99    113732
           1       0.06      0.92      0.11       191

    accuracy                           0.97    113923
   macro avg       0.53      0.95      0.55    113923
weighted avg       1.00      0.97      0.99    113923

```

#### **Class Weights**

Also tried it with `Class Weights`, it gives more weightage to the predictions of minority class.

The Output was:
```
Using Class Weights ... 
Creating our Model ... 
Fitting the model ... 
Model training completed.
Confusion Matrix:
[[111032   2700]
 [    17    174]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99    113732
           1       0.06      0.91      0.11       191

    accuracy                           0.98    113923
   macro avg       0.53      0.94      0.55    113923
weighted avg       1.00      0.98      0.99    113923

```

It can be observed that all 3 methods of handling data imbalance (`RandomOverSampler`, `SMOTE`, `Class Weights`) gave similar results. They reduced the number of frauds going undetected (`False Positive`), but also drastically increased the numbers of falsely detected frauds (`False Negative`). I am assuming this is because the dataset is highly imbalanced, it only has ~0.17% for the minority class,