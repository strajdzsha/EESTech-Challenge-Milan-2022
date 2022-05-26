# EESTech-Challenge-Milan-2022

Code for our solution of EEStech Challenge Milan 2022 problem.

To run the code in real time, start script "code\live_testing". Model can be imported from file "model_new.joblib" and you will also need to import scaler from "scaler_new.joblib" as well as PCA model from "pca.joblib".

"data.csv" contains data frame used for training the model. It contains around 15000 frames of recording with extracted features which are fed into XGBoost model.

Quick summary of all modules found in "code":
Modules used for internal purposes (you wont need them actually):
- creat_train_data.py
- data_collecting.py
- fft.py
- processing.py
- labeling_data.py

Modules you can run to train/test the model, or to visualise data in animation:
- model.py
- visualising_data.py
- training_model.py
- live_testing.py
