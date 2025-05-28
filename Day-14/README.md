## Day 14

### Gradient Boosting Algorithms

Learned about the theory behind `Gradient Boosting` and implemented `GradientBoostingClassifier` and `GradientBoostingRegressor` from `Scikit-Learn`.

Implemented the `GradientBoostingClassifier` on the `breast cancer dataset` which I had previously used with `Logistic Regression`. 
I had got an accuracy of `97%` accuracy with `Logistic Regression`, but got `95.6%` accuracy with `GradientBoosting`.

Implemented the `GradientBoostingRegressor` on the `wine quality dataset` which I had previously used with `Linear Regression` for Regression and with `Random Forest` for classification.
I rounded off the values predicted by `GradientBoostingRegressor` to use it for classification as well.
I had previously gotten an accuracy of almost `65.6%` with `Random Forest`, but now I got an accuracy of `57.2%`.

`Gradient Boosting` performed worse than `Logistic Regression` and `Random Forest` for classifcation, but for Regression with `Linear Regression` I had previously gotten Mean Squared Error of around `0.388301`, but with `GradientBoostingRegressor` the Mean Squared Error was `0.367384`.

So it did performed well with `Regression` but well with `Classification`.