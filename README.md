# Parkinson's-Disease-Detection

Parkinson's Disease is a progressive neuorological disorder that affects movement and coordination. It occurs due to the loss of dopamine-producing neurons in the part of the brain alled substania nigra.
Common symptoms include speech defects, tremors, slowness of movement, stiffness and balance problems over time.

# Overview

Machine Learning Model SVM ( Support Vector Machine ) was used in this case for the prediction of parkinson's disease. Model was trained on 195 datasets (only available).
Feature Extraction like MDVP features, jitter, shimmer etc. were made.
Accuracy of the model achieved was to be 87.17%.

Intially the model was taking separate mannual input values from the user which was collected after performing of different speech examination and medical tests, this model was created / updated to take both comma seperated values for each column mannually and for to automatically extract the necessary features required for the prediction of disease from the voice.


The model takes voice as input, extracts the various components of speech from the voice which are used passed to the model trained usiing svm on a dataset containing the 22 features of speech samples of different people both affected and unaffected with parkinson's disease.

Further the model was integreated with a webapp created using flask (python) used to provide an interface to the user to test for the prediction of prakinson's disease in a much simpler way.
Increasing the feasiblity of the model and the app.

# MODEL
A machine learning-based web application that predicts the presence of Parkinson’s Disease from voice data. This project combines data science, machine learning (SVM), and Flask-based web development to provide a complete solution for early detection of Parkinson’s Disease through vocal biometrics.


# Model Architecture
 - Algorithm: Support Vector Machine (SVM) with a linear kernel.
 - Preprocessing:
    - Removal of irrelevant columns (name.status)
    - Standardization using StandardScaler
-Accuracy: Achieved an accuracy of 87.17% on the dataset.

# Web Appilcation 

A responsive and user-friendly web interface was developed using Flask to make predictions in two modes:

1. Comma-Separated Input: Paste all 22 features in one line.
   
2.   Individual Feature Input: Manually enter each feature separately.

Prediction result is shown instantly on submission.

# Features
- Voice-based detection using speech biomarkers
- SVM-based predictive modeling
- Dual input interface (CSV-style and manual)
- Flask-powered backend
- Responsive HTML interface for better UX

# Getting Started

**1. Clone the repository:**
*git clone https://github.com/rishavkumar584/Parkinson-s-Disease-Detection*

*cd Parkinsons-Disease-Detection*

**2. Install dependencies:**

**3. Start the app**

On your local server *python app.py*

Open your browser and go to http://127.0.0.1:5000

- Choose either input method:
-  - Enter 22 comma-separated values
   - Fill individual input fields
   - Upload the path of your audio file in the python code
     @ "audio_file = "Maire_Positive.wav"  # Replace with your audio file path here"
- Click Predict
- See the result displayed on screen

# Result
- Training Accuracy: 87.17%
- Robust to small datasets
- Real-time inference on user input

  # Technologies Used:
  - Python
  - Scikit-learn
  - Pandas, NumPy
  - Flask (Backend)
  - HTML/CSS (Frontend)
  - Jupyter Notebook (Exploratory Data Analysis)
 
  # Conlusion

  Developed with dedication for real-world healthcare impact project demonstration.
