# MCU_active_learning_for_face_liveness_detection


## gather_examples.py

## divide_file_randomly.py

## change_pic_size.py

The above three files can be opened using any Python compiler to obtain face data from the uploaded video through the command line:

Real face data:

$ python gather_examples.py --input videos/real.mov --output dataset/real \ --detector face_detector --skip 1 (take the number of frames in the data interval)

Fake face data:

$ python gather_examples.py --input videos/fake.mov --output dataset/fake \--detector face_detector --skip 4 (take the number of frames in the data interval)

Then by modifying the folder location in change_pic_size.py and running, the obtained face data can be converted into the same size (length * width).
Finally, by modifying the folder location in divide_file_randomly.py and running, the data can be randomly divided into 7:3 training set and test set.

## active_learning_face_liveness.ipynb

This notebook contains the CNN model and the active learning model, run this notebook can get the model and the accuracy after 450 epochs.
