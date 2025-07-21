# 📝 Handwritten Digit Classifier (Streamlit App)

This is a simple web application built with **Streamlit** that demonstrates handwritten digit classification using a **Decision Tree** classifier.  
The classifier predicts the digit label from an existing CSV file (`handwriting.csv`) that contains pre-processed, flattened 28×28 grayscale images — similar to MNIST.

---

## 📌 Features

- 📂 Load image data from a CSV file
- 🔢 Choose an image index to view the digit
- 🧠 Predict digit label using a trained Decision Tree classifier
- 🖼️ Display image in compact size (via `st.image()` or `matplotlib`)
- ✅ Clean UI with proper error handling and interactivity

---

## 📁 Dataset Format

The dataset should be a `.csv` file structured like this:

- **First column**: `label` (the actual digit: 0–9)
- **Remaining columns**: pixel values of the 28×28 image, flattened into 784 columns

Example (first few columns):

label,pixel1,pixel2,...,pixel784
5,0,0,...,255

---

## 🚀 How to Run

### 1. Clone the Repository

```bash

git clone https://github.com/yourusername/handwritten-digit-classifier.git
cd handwritten-digit-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Files

- Place your CSV file as handwriting.csv in the root directory.

- Add a trained Decision Tree model saved as model.pkl (optional if you're using one).

### 4. Run the App

```bash

streamlit run app.py

```

## 🔗 Live Demo

👉 [Click here to try it out on Streamlit](https://devraaz-handwritten-digit-classifier.streamlit.app/)

---

## 💻 GitHub Repo

🔗 [View Full Source Code on GitHub](https://github.com/Devraaz/Handwritten-Digit-Classifier)

---

## 🖥️ App UI

### ✅ Main Interface:

- Input an index (e.g., 0 to 9999)

- View the digit as an image

- Click "Detect Number" to get the predicted label

### 📊 Display Options:

- Compact image rendering using st.image() with pixel-width control

- Optional matplotlib visualization (st.pyplot) with adjustable figsize

## 🧠 Model Used

### Model: Decision Tree Classifier (sklearn.tree.DecisionTreeClassifier)

- Trained On: Flattened grayscale handwritten digits from the CSV

- Goal: Demonstrate simple classification logic (not production-ready ML)

## 🛠️ Tech Stack

- 🐍 Python 3.x

- 🧠 scikit-learn

- 📊 pandas, matplotlib, numpy

- 🌐 Streamlit (for UI)

## 📘 Notebook Included

This repository includes a **Jupyter Notebook** that shows:

- Full data cleaning and exploration steps
- Model training and evaluation
- Accuracy metrics and confusion matrix

---

## 🙋‍♂️ Author

Made with ❤️ by [Devraj Dora](https://github.com/devraaz)
