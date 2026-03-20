# ML-Practice

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

Welcome to my personal Machine Learning and Deep Learning hub.

This repository is **not intended as** a tutorial or a course.
This is just a **learning playground**, where I implement and experiment with ML concepts, and occasionally try Deep Learning architectures.

---

## Topics:

### Data Analysis and Preprocessing
- [x] Basic Data Pre-processing
- [x] Explanatory Data Analysis
- [x] Feature Engineering

### Machine Learning
- [x] Linear Regression
- [x] Logistic Regression
- [x] K-Nearest-Neighbours
- [x] Decision Trees
- [x] Random Forest
- [x] K-Means Clustering
- [x] Support Vector Machines
- [x] Gradient Boosting Algorithms
- [x] Hyperparameter Tuning
- [x] Model Evaluation
- [x] Data Imbalance Handling
- [ ] Pipelines

### Deep Learning
- [x] Neural Networks (Perceptrons)
- [x] Convolutional Neural Networks (CNN)
- [x] Recurrent Neural Networks (RNN)
- [x] Gated Recurrent Units (GRU)
- [x] Long Short-term Memory (LSTM)
- [ ] Transformers

### Long-term Plan
- [ ] Natural Language Processing
- [ ] Computer Vision

---

## Dataset Index

| Dataset | Source | Used In |
|---------|--------|---------|
| Titanic Dataset | [Kaggle](https://www.kaggle.com/competitions/titanic/data) | Day-02, Day-18 |
| Student Scores | [Kaggle](https://www.kaggle.com/datasets/mexwell/student-scores) | Day-03, Day-12 |
| Breast Cancer Dataset | [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) | Day-04, Day-09, Day-13 |
| Diabetes Dataset | [GitHub](https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/blob/master/diabetes.csv) | Day-05, Day-10, Day-17 |
| Iris Dataset | [UCI](https://archive.ics.uci.edu/dataset/53/iris) | Day-06 |
| MNIST | [HuggingFace](https://huggingface.co/datasets/ylecun/mnist) | Day-01, Day-07, Day-14 |
| Wine Quality (Red) | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) | Day-08, Day-13 |
| Ames Housing Dataset | [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) | Day-12 |
| Credit Card Fraud Dataset | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Day-11 |
| Tiny Shakespeare | [GitHub](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) | Day-15, Day-16 |

---

## Structure

Each `Day-XX-*` directory represents a concept/topic. Inside, you’ll find:
- `.py` scripts implementing the concept.
- Output plots or visualizations *(when applicable)*
- Sample datasets *(if required)*
- A `requirements.txt` file, to install the required dependencies.
- A per-topic `README.md` that gives a quick summary.

## Environment

If you want to try out any implementation in this repository, you can,
- Clone the repository,
  ``` bash
  git clone https://github.com/IshaqJunejo/ML-Practice.git
  ```
- Navigate to your concerned implementation, and make sure you have installed dependencies using `pip install -r requirements.txt`, preferably in a virtual environment.
- And run the code.

***Note**: Each `Day-XX-*` directory may use different Python packages. Refer to the imports in the individual scripts.*

## Notes

- This is mostly self-supervised learning from multiple sources (YouTube, and ChatGPT suggestions).
- `.py` scripts are used instead of `.ipynb` notebooks due to personal preferences.
- The number of `Days` seen in the directory naming may be misleading. Some "Day"s took longer than a day, and some came after a lazy break.

## License

This is a personal practice laboratory, so feel free to use for your own learning.
