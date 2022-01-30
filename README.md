# Live-Class-Monitoring-System-Face-Emotion-Recognition-

## Introduction

Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context. To date, the most work has been conducted on automating the recognition of facial expressions from video, spoken expressions from audio, written expressions from text, and physiology as measured by wearables.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.

![image](https://user-images.githubusercontent.com/87691447/151480962-f9fb2bce-d88e-4f01-972c-9ac5fd787ca9.png)


## Problem Statements

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

## What is Face Emotion Recognition?

- Facial emotion recognition is the process of detecting human emotions from facial expressions.
- The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well.
- This is a few shot learning live face emotion detection system.
- The model should be able to real-time identify the emotions of students in a live class.

## Head-start References

❖ https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lin es-of-code-b2ce32d4d5de

❖ https://towardsdatascience.com/video-facial-expression-detection-with-deep-learning-appl ying-fast-ai-d9dcfd5bcf10

❖ https://github.com/atulapra/Emotion-detection

❖ https://medium.com/analytics-vidhya/building-a-real-time-emotion-detector-towards-machi ne-with-e-q-c20b17f89220

## Dataset link 
https://www.kaggle.com/msambare/fer2013

## Dataset Information

The data comes from the past Kaggle competition “Challenges in Representation Learning: Facial Expression Recognition Challenge”. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

This dataset contains 35887 grayscale 48x48 pixel face images.

Each image corresponds to a facial expression in one of seven categories

Labels:

0 - Angry 😠

1 - Disgust 😧

2 - Fear 😨

3 - Happy 😃

4 - Sad 😞

5 - Surprise 😮

6 - Neutral 😐

## Dependencies

Python 3
Tensorflow 2.0
Streamlit
Streamlit-Webrtc
OpenCV

## Project Approch

#### Step 1. Build Model

We going to use three different models including pre-trained and Custom Models as follows:

 - Method1: Using DeepFace
 - Method2: Using ResNet
 - Method3: Custom CNN Model
 - 
#### Step 2. Real Time Prediction

And then we perform Real Time Prediction on our best model using webcam on Google colab itself.

  - Run webcam on Google Colab
  - Load our best Model
  - Real Time prediction
  - 
#### Step 3. Deployment

And lastly we will deploy it on three different plateform

   -  Streamlit Share
   -  Heroku
   -  Amazon WEb Services (AWS)

![image](https://user-images.githubusercontent.com/87691447/151481439-8259f8cc-bb59-4f2e-9522-23664d4e6b17.png)

## Method 1: Using DeepFace

Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python.

It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib.

## Method 2: Using ResNet

![image](https://user-images.githubusercontent.com/87691447/151481607-503ddd35-d61a-4d5b-acd9-5dfd1c101669.png)

ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks.

This model was the winner of ImageNet challenge in 2015. The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully.

ResNet makes it possible to train up to hundreds or even thousands of layers and still achieves compelling performance.

![image](https://user-images.githubusercontent.com/87691447/151688512-29659852-b98e-4d0c-b7de-65ba23633e1d.png)

![image](https://user-images.githubusercontent.com/87691447/151688520-7f868eaa-5bd4-4460-b0a1-5cbf96aabf55.png)


Plotting Accuracy & Loss to analyze the results:

![image](https://user-images.githubusercontent.com/87691447/151481835-cb867f63-6c98-438a-bd3d-65d6fe304675.png)

![image](https://user-images.githubusercontent.com/87691447/151481851-284e34b2-9e30-4dd0-9308-13b6e7f857f2.png)

## Method 3: Custom CNN Model

Convolutional Neural Network, also known as CNN is a sub field of deep learning which is mostly used for analysis of visual imagery.

It is composed of multiple layers of artificial neurons.

Artificial neurons, a rough imitation of their biological counterparts, are mathematical functions that calculate the weighted sum of multiple inputs and outputs an activation value.

![image](https://user-images.githubusercontent.com/87691447/151506674-72656401-6cdb-4462-8953-ab1fd859c9cf.png)


Plotting Accuracy & Loss to analyze the results:

![image](https://user-images.githubusercontent.com/87691447/151481990-0c6a9175-83d7-467f-b88d-a007e3cfd191.png)

The training gave the accuracy of 74% and val_accuracy of 60%. It seems good. So, I save the model and detection I got from live video is good.
The training loss is slightly higher than the validation loss for the first epochs.

## Confusion matrix

Finally we can plot the confusion matrix in order to see how our model classified the images:

![image](https://user-images.githubusercontent.com/87691447/151482068-097f973c-bf9b-4169-9151-2db21534745e.png)

Our model is very good for predicting happy and surprised faces. However it predicts quite poorly feared faces maybe because it confuses them with sad faces.

## Deployment of Streamlit WebApp in Heroku and Streamlit

We have created front-end using Streamlit for webapp and used streamlit-webrtc which helped to deal with real-time video streams. Image captured from the webcam is sent to VideoTransformer function to detect the emotion. Then this model was deployed on heroku platform with the help of buildpack-apt which is necessary to deploy opencv model on heroku.

Streamlit: https://share.streamlit.io/codewithsridhar/live-class-monitoring-system-face-emotion-recognition-/main/app.py

Heroku: https://face-emotion22.herokuapp.com/ 

## Conclusion

We trained the neural network and we achieved the highest validation accuracy of 60.43%.

Pre Trained Model didn't gave appropriate result.

The application is able to detect face location and predict the right expression while checking it on a local webcam.

The front-end of the model was made using streamlit for webapp and running well on local webapp link.

Successfully deployed the Streamlit WebApp on Heroku and Streamlit share that runs on a web server.

Our Model can succesfully detect face and predict emotion on live video feed as well as on an image.
