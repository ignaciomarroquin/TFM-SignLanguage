# TFM-SignLanguage
Real-time Spanish Sign Language detection using TensorFlow, MediaPipe, and a custom trained CNNs. This project was developed as my Master's Final Thesis (TFM).

This repository contains all the code used throughout the thesis, as well as a demonstration video showcasing the functionality of the real-time web interface.

Repository Structure
implementacion/: Contains the code related to the development of the web interface, including the HTML layout and example images.

keypoints.ipynb: A Jupyter Notebook demonstrating how additional datasets were generated from the original dataset. It also includes the train/validation/test data split process.

main_code_TFM.ipynb and forlongtraings.py: These are equivalent scripts (in notebook and .py formats). They represent the core training pipeline used for running all model training experiments.

evaluarbestmodels.py: A helper script designed to generate cleaner confusion matrices and verify that the models saved during training (from forlongtraings.py) were indeed the best-performing ones. It also checks that the metrics tracked via Neptune correspond to the correct model checkpoints.
