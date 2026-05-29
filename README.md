#  🍳 FOOD Classification 

> **About this project**
> This project was developed as the final assignment for the Advanced Machine Learning course taught by Professor Nwe Nwe Htay Win. The goal of the project is to classify images across 101 different food categories using deep learning and transfer learning techinques.

This repository covers the complete machine learning workflow -- from dataset preparation and model training to deployment through an interactive [web application](https://foodddd-ftfqhgejajahhcen.japaneast-01.azurewebsites.net) hosted on Microsoft Azure.

---

## 📱 The Live Web Application

### 1. Upload Interface
<img src="Images/webapp_ui.png" alt="Clean, drag-and-drop web portal" width="100%">

### 2. Real-Time Model Prediction
<img src="Images/webapp_prediction.png" alt="Top-5 dish breakdown with confidence scores" width="100%">

> **How to use the web app**
> Upload an image from your device and click **Analyze Image** to receive predictions from the model!
> You can also copy an image directly from the internet and paste it into the upload area, and analyze it instantly!

---

## 📁 Repository Structure

* `app/`: Contains the backend Flask application (`app.py`), HTML templates (`templates/`), and the upload directory (`static/uploads`) for temporarily storing uploaded images.
* `Code Notebook/`: Includes the step-by-step Jupyter Notebook used for model training and evaluation.
* `presentation/`: Presentation slides used during the final project presentation.
* `guides_&_docs/`: Additional documentation and resources created for students interested in CNNs and transfer learning:
  * 📄 `annotated_references.md`
     A curated list of references, YouTube videos and learning resources used throughout the project, along with personal notes and summaries..
  * ⏳ `food101_code_walkthrough.pdf` *(Coming in a few days!)*
    A detailed walkthrough explaining the project's code structure, implementation choices, and core CNN + tranfser learning concepts in a beginner-friendly manner
  * ⏳ `food101_project_report.pdf` *(Coming in a few days!)*
    A formal research-style report covering the project's motivation, methodology, experiments, and results.
