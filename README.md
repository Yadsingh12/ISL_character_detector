# ISL_character_detector
I made this project to make a live detection software that detects english characters as represented in Indian Sign language.


DATASET:

the dataset consists of various resources bundled together:

https://data.mendeley.com/datasets/7tsw22y96w/1

https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset


MODEL PERFORMANCE:

Although the earlier versions (versions_1 and version_2) works well on the dataset(80 percent accuracy for version_1 and 90% accuracy for version_2 on validation dataset), yet they still do poorly on live detection. They show big signs of bias where a handful of signs are classified much more than others.


PERSONAL HISTORY:


START OF THE JOURNEY:

I started this project as part of my summer training at Punjabi University, Patiala. At the start, I alread had some knowledge on Indian sign language and I had just started with learning Convolutional neural networks. I wanted to try the cool sounding CNNs so I started with this project. Earlier, I though that it would be just as simple as taking a dataset from internet and training the model. Howwever as the days passed more and more complexities started to arise.


USING MOBILENETV2 AND GOOGLE COLAB NOTEBOOKS:

I used google's mobilenetV2 as the base for the model and tried to classify signs from the images. I did not have a gpu, so I used google colab for training the model. It took around 6 hourse for the first time I trained the model. Then, I used a callback function that can store the model when a training epoch is completed. This made it possible for me to train the model in chunks instead of training it in one go and lossing all the training if the server disconnected.


WHEN NUMBERS NOT CONVEY EVERYTHING:

I was exciting to see my model gaining an accuacy of 80 percent on validation dataset. However, I did not noticed that since, the model was not trained in one go, the random split between training and validation dataset would be different each time. This meant that when I retrain the model at a different time, it had already been trained on most of the validation dataset.