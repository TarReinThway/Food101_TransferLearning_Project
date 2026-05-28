#  🍳 FOOD Classification 

> **What is this project?** This is the final project of Advanced Machine Learning Course taught by Professor Nwe Nwe Htay Win. It is an image classification project meant to classify between 101 different type of foods.

This repository covers the entire data science lifecycle: from raw images to a deep learning model and an interactive [web application](https://foodddd-ftfqhgejajahhcen.japaneast-01.azurewebsites.net) hosted on Microsoft Azure.

---

## 📱 The Live Web Application

### 1. Upload Interface
<img src="Images/webapp_ui.png" alt="Clean, drag-and-drop web portal" width="100%">

### 2. Real-Time Model Prediction
<img src="Images/webapp_prediction.jpg" alt="Top-5 dish breakdown with confidence scores" width="100%">

> **How to use this webapp?** Just choose the image from your file, click analyze and get results! Alternatively, if you are on a browser, you can copy any image online, hover mouse over the box and paste the image, then analyze to get the results!

---

## 📁 Repository Structure

* `app/`: Contains the backend Flask application (`app.py`), HTML interfaces (`templates/`), and the upload directory (`static/uploads`) for temporarily caching user images.
* `Code Notebook/`: The step-by-step Jupyter Notebook where the model was trained and evaluated.
* `presentation/`: Slides used during the project evaluation and presentation. 
* `guides_&_docs/`: In-depth educational documentation curated specifically for SDS juniors:
  * 📄 `annotated_references.md` - A hand-picked literature list detailing the academic resources that shaped this project, complete with personal notes.
  * ⏳ `food101_code_walkthrough.pdf` *(Coming in a few days!)* - A detailed, informal guide breaking down every line of code, logical reasoning, and core CNN + Transfer Learning concepts—tailored specifically for students who have only taken an introduction to Python.
  * ⏳ `food101_project_report.pdf` *(Coming in a few days!)* - A formal, research-style paper detailing the project's background, technical methodology, and experimental results.
