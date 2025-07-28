## Day 18 - Feature Engineering

Implemented Survivor prediction on the `Titanic Dataset` using multiple models (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Support Vector Machine`, and `K Nearest Neighbours`).

Did the implementation with and without some basic **Feature Engineering** to analyze the impact it makes on the performance on the prediction.

### Before
Only basic data pre-processing was applied for testing without Feature Engineering.

This data pre-processing consisted of:
- Dropping irrelevant columns.
- Filling missing values.
- Label Encoding Categorical values.
- Standard Scaling values.

```
Logistic Regression ...
Accuracy:  0.8100558659217877

Confusion Matrix:
[[90 15]
 [19 55]]

F1 Score:  0.7638888888888888
```

```
Decision Tree ...
Accuracy:  0.7821229050279329

Confusion Matrix:
[[83 22]
 [17 57]]

F1 Score:  0.7450980392156863
```

```
Random Forest ...
Accuracy:  0.8044692737430168

Confusion Matrix:
[[88 17]
 [18 56]]

F1 Score:  0.7619047619047619
```

```
SVM ...
Accuracy:  0.8156424581005587

Confusion Matrix:
[[93 12]
 [21 53]]

F1 Score:  0.762589928057554
```

```
K Nearest Neighbours ...
Accuracy:  0.8044692737430168

Confusion Matrix:
[[90 15]
 [20 54]]

F1 Score:  0.7552447552447552
```


### After
For testing with Feature Engineering, some basic data pre-processing (as mentioned previously) was applied with some basic Feature Engineering.

The Feature Engineering consisted of:
- Extracting "Titles" from "Names".
- Creating a column for "Family Size" from "Siblings / Spouse" and "Parents / Children" columns.
- Making Bins for "Age" and "Fare", to help mitigate the outlier effect.
- Getting "Deck" as the first character in "Cabin" (most are just Unknown).
- Label Encoding "Gender", and One-hot Encoding "Embarked City", "Title", and "Deck".

```
Logistic Regression ...
Accuracy:  0.8212290502793296

Confusion Matrix:
[[89 16]
 [16 58]]

F1 Score:  0.7837837837837838
```

```
Decision Tree ...
Accuracy:  0.770949720670391

Confusion Matrix:
[[85 20]
 [21 53]]

F1 Score:  0.7210884353741497
```

```
Random Forest ...
Accuracy:  0.8100558659217877

Confusion Matrix:
[[87 18]
 [16 58]]

F1 Score:  0.7733333333333333
```

```
SVM ...
Accuracy:  0.8212290502793296

Confusion Matrix:
[[88 17]
 [15 59]]

F1 Score:  0.7866666666666666
```

```
K Nearest Neighbours ...
Accuracy:  0.7877094972067039

Confusion Matrix:
[[87 18]
 [20 54]]

F1 Score:  0.7397260273972602
```

### Performance Summary

This is the summary of performance of each model with and without **Feature Engineering** based on **F1 Score**.

| Model               | Without FE | With FE |
|---------------------|------------|---------|
| Logistic Regression | 0.763      | 0.783   |
| Decision Tree       | 0.745      | 0.721   |
| Random Forest       | 0.761      | 0.773   |
| SVM                 | 0.762      | 0.786   |
| KNN                 | 0.755      | 0.739   |

Overall, other than `Decision Tree` and `KNN`, models did perform slightly better with Feature Engineering as compared to without Feature Engineering.

I was expecting the performance gain to much higher based on what I have heard about it, but this is also acceptable for the sake of practice.
