# Replicate_Facial_Expressions_of_Emoji
The model is already trained on the dataset and stored in the emotion_model.h5 file.

# Steps to run the application
1. unzip the data.zip folder
2. delete the emotion_model.h5 (only if you want to re-train the model)
3. run the gui.py file for the graphical user interface

# Commands for the steps mentioned above
Train the model:  python train.py
This command will give the output of the model summary and train the model with the dataset given upto 10 epochs.

Run the UI: python gui.py
This command will start the camera and open the application with the face in the camera and emoji.
