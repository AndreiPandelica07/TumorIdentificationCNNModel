# TumorIdentificationModel
This repository contains my personal project working with an open kagglehub dataset containing images of different brain scans, structured by type of image ( tumor type/ no tumor). The model consists of a convolutional network implemented using the PyTorch library from scratch.

The current model presents a recall of ~88%, given that it currently struggles identifying all types of tumors, sometimes mismatching one of the 3. 

However, given the current amount of data, the current stats of the model hold up high. For the future I am looking forward to finding bigger datasets which present information neccesarry to our model's current predicition outputs.

The current arhitecture has been experimented with, along with the input sizes, finding that for the structures tested the recall and overall accuracy doesn't depend much on it ( at least based on what I've experimented with ).

The current best model is saved within "cnn_model2.pth", along with the best training results inside "average_lossBest_Model2.pth".