# 🎓 Final Year Project: Chronic Kidney Disease (CKD) Prediction using Machine Learning


## 💡 Project Overview & Motivation

**Chronic Kidney Disease (CKD)** is a severe health condition where the kidneys are damaged and lose their ability to effectively filter waste and excess fluid from the blood. Since CKD often develops **slowly and without noticeable symptoms** in its early stages, detection can be highly challenging, leading to delayed treatment and increased health risks.

This project addresses this critical challenge by leveraging **Machine Learning (ML) techniques** to build an accurate predictive model for CKD. Our goal is to aid in **early detection**, allowing medical practitioners to intervene sooner, which can drastically improve patient outcomes.


## 🎯 Scope and Objectives

The core of this work involves applying various supervised machine learning and deep learning algorithms to predict the presence of Chronic Kidney Disease.

* **Dataset:** Utilized a publicly available dataset from the **UCI Machine Learning Repository**, comprising **400 samples**, **24 input features**, and **1 target variable** (CKD/Not-CKD).
* **Predictive Modeling:** Compile and compare a set of high-performance prediction models to establish the most effective method for this specific dataset.
* **Advanced Goals:** The project also explored avenues for **performance improvement** and the feasibility of implementing an **advanced chatbot** for user assistance and a **deep learning model** for robust disease prediction.



## ⚙️ Project Workflow

Our systematic approach to developing and validating the predictive models followed these key steps:

1.  **Data Preparation:** Collect the raw data, perform initial cleaning, and handle missing values.
2.  **Explore & Prepare:** Conduct in-depth data analysis (EDA), and perform feature engineering/selection where necessary.
3.  **Split Data:** Divide the prepared dataset into distinct **training** and **testing** sets.
4.  **Model Selection:** Choose and configure a diverse set of machine learning and deep learning models.
5.  **Train the Model:** Fit each model to the training data.
6.  **Evaluate:** Rigorously test the models’ performance on the unseen test data using metrics like Accuracy, Precision, Recall, and F1-score.
7.  **Refine:** Tune the model’s **hyperparameters** to maximize performance.
8.  **Validate:** Ensure that performance improvements generalize reliably through techniques like cross-validation.
9.  **Deploy (Future Step):** Prepare the final, best-performing model for potential real-world use or simulation.


## 🧠 Machine Learning Algorithms Used

A comprehensive set of classical machine learning classifiers, known for their strong performance in tabular data classification, were implemented and compared:

* **K-Nearest Neighbors (KNN)**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Gradient Boosting Classifier**
* **AdaBoost Classifier**
* **XGBoost**
* **CatBoost Classifier**


## ⚠️ Dataset Overview & Limitations

The investigation utilized a specific dataset, which inherently introduced some limitations that must be considered when interpreting the results:

1.  **Small Sample Size:** The dataset contains only **400 samples**. This small size is a significant constraint on the reliability and generalizability of the analysis, as larger datasets typically yield more robust and universally applicable results.
2.  **Performance Evaluation Bias:** The use of an additional, specific dataset for problem identification and performance evaluation may restrict the model's ability to represent a wide distribution of real-world data, potentially affecting the accuracy of cross-validation results.


## ✅ Key Results and Conclusion

Our study successfully developed highly accurate predictive models for Chronic Kidney Disease. Comparing our results with previous literature:
**XGBoost** : 98.3%  
**AdaBoost Classifier** : **99.1%**  (**Highest accuracy achieved in this study.**)

The **AdaBoost Classifier** achieved an accuracy of **99.1%**, surpassing the previously reported highest accuracy of 99.0% achieved with Gradient Boosting on similar publicly available CKD datasets. 

This project offers a significant contribution to the medical community's understanding of applying advanced machine learning techniques for early and accurate disease prediction.
