#  🍳 FOOD Classification 

> **What is this project?** This is the final project of Advanced Machine Learning Course taught by Professor Nwe Nwe Htay Win. It is an image classification project meant to classify between 101 different type of foods.

This repository covers the entire data science lifecycle: from raw images to a deep learning model and an interactive [web application](https://foodddd-ftfqhgejajahhcen.japaneast-01.azurewebsites.net) hosted on Microsoft Azure.

---

## 📱 The Live Web Application

| 1. Upload Interface | 2. Real-Time Model Prediction |
| :---: | :---: |
| <img src="Images/webapp_ui.png" width="400"> | <img src="Images/webapp_prediction.png" width="220"> |
| *Clean, drag-and-drop web portal.* | *Top-5 dish breakdown with confidence scores.* |

> **How to use this webapp?** Just choose the image from your file, click analyze and get results! Alternatively, if you are on a browser, you can copy any image online, hover mouse over the box and paste the image, then analyze to get the results!

---

## 📁 Repository Structure

* `app/`: Contains the backend Flask application (`app.py`), HTML interfaces (`templates/`), and upload folder for storing uploaded images (`static/uploads`)
* `Code Notebook/`: The step-by-step Jupyter Notebook where the model was trained and evaluated.
* `guides_&_docs/`: In-depth educational guides, including an `annotated_references.md` where the resouces that were helpful to me in writing this project is shared with annotations. In a few days, `food101_code_walkthrough.pdf` where I will be explaning every code, their logic & reasoning as well as CNN + Transfer Learning concepts in details with SDS students who have taken introduction to python in mind, as well as `food101_project_report.pdf` where I will upload a report of the project. 
* `presentation/`: Slides used during presentation. 
