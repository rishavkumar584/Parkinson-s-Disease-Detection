# Parkinson-s-Disease-Detection

Parkinson's Disease is a progressive neuorological disorder that affects movement and coordination. It occurs due to the loss of dopamine-producing neurons in the part of the brain alled substania nigra.
Common symptoms include speech defects, tremors, slowness of movement, stiffness and balance problems over time.

# Description

Machine Learning Model SVM ( Support Vector Machine ) was used in this case for the prediction of parkinson's disease. Model was trained on 195 datasets (only available).
Feature Extraction like MDVP features, jitter, shimmer etc. were made.
Accuracy of the model achieved was to be 87.17%.

Intially the model was taking separate mannual input values from the user which was collected after performing of different speech examination and medical tests, this model was created / updated to take both comma seperated values for each column mannually and for to automatically extract the necessary features required for the prediction of disease from the voice.

The model takes voice as input, extracts the various components of speech from the voice which are used passed to the model trained usiing svm on a dataset containing the 22 features of speech samples of different people both affected and unaffected with parkinson's disease.

Further the model was integreated with a webapp created using flask (python) used to provide an interface to the user to test for the prediction of prakinson's disease in a much simpler way.
Increasing the feasiblity of the model and the app.
