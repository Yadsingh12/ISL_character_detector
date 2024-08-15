# ISL_character_detector
I made this project to make a live detection software that detects english characters as represented in Indian Sign language.


DATASET:

the dataset consists of various resources bundled together:

https://data.mendeley.com/datasets/7tsw22y96w/1

https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset


AIM:
The goal of the project is to build a reliable detection system that can detect the representation of engish character as in Indian sign language in a live detection environment with a fairly good accuracy.

MODEL PERFORMANCE:

Although the earlier versions (versions_1 and version_2) works well on the dataset(80 percent accuracy for version_1 and 90% accuracy for version_2 on validation dataset), yet they still do poorly on live detection. They show big signs of bias where a handful of signs are classified much more than others.


PERSONAL HISTORY:


START OF THE JOURNEY:

I started this project as part of my summer training at Punjabi University, Patiala. At the start, I alread had some knowledge on Indian sign language and I had just started with learning Convolutional neural networks. I wanted to try the cool sounding CNNs so I started with this project. Earlier, I though that it would be just as simple as taking a dataset from internet and training the model. Howwever as the days passed more and more complexities started to arise.


USING MOBILENETV2 AND GOOGLE COLAB NOTEBOOKS:

I used google's mobilenetV2 as the base for the model and tried to classify signs from the images. I did not have a gpu, so I used google colab for training the model. It took around 6 hourse for the first time I trained the model. Then, I used a callback function that can store the model when a training epoch is completed. This made it possible for me to train the model in chunks instead of training it in one go and lossing all the training if the server disconnected.


WHEN NUMBERS NOT CONVEY EVERYTHING:

Since, I was short on data, I chose not to have a seperate tarining dataset. I was exciting to see my model gaining an accuacy of 80 percent on validation dataset. However, I did not noticed that since, the model was not trained in one go, the random split between training and validation dataset would be different each time. This meant that when I retrain the model at a different time, it had already been trained on most of the validation dataset.

WRONG WAY OF DOING AUGMENTATION:

The way the I implemented the augmentation in my code made the images of the signs vastly different from what they originally meant to be. Due to this, it became very difficult for the model to distinguish between some signs.('A' & 'B', 'C' & 'O', 'U' & 'V', 'E' & 'J' etc.). So, I had to redo the augmentation.


RETRAINING THE MODEL:

After being upset about my model's poor performance when used for live detection, I chose to retrain my model. I thought maybe more training would be beneficial. After training, the newer model (version_2) got a 90 percent accuracy on validation dataset and an increase in performance in live detection but it was still biased and would output a haandful of characters('B' 'C' 'D' 'G') in place of the correct labels most of the time in live detection.


REASON FOR POOR PERFORMANCE:

The biggest reason for the poor performance of the model was that the the background of the images include a lot of noise ( colors, clothes, skin, faces etc.). Though I intentionally included a dataset with a lot of noise so that the model can classify in a noisy environment like live detection but this also meant that model learnt wrong patterns like colors and background of images instead of the position of hands while classifying the signs.


USING MEDIAPIPE FOR HAND DETECTION:

In order to battle the noise in the bcakground, I started doing research. Then I came across a research paper that used openCV to detect ASL signs through camera. One thing they used was to first use mediapipe to detect hands from the data and then extract those hands, apply white background to the images and then use CNN and sequential model for predictions.