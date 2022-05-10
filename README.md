## **Project Introduction**

In this project I'll be using the UTKFace dataset which has around ~24,000 frontal face images to build a face detector with gender and age prediction web-app. I'll also be using transfer learning with pre-trained model **DenseNet121** to make better regression and classification models for our use case. To optimize performance, I'll try to tune and experiment with some hyper-parameters. Finally, I'll save the models and pack everything into a web-app using Streamlit and host it to cloud for anyone to use.

**WEB-APP LINK:** [Face Detection with Age and Gender Prediction](https://share.streamlit.io/rishi5565/face-detection-with-age-and-gender-prediction/main/app.py)


## **Project Overview**

* Built a frontal face detection with gender and age prediction web-app from scratch on images taken from upload or camera and deployed to cloud for anyone to use.
* Used Streamlit to build front-end interface of the web-app.
* Cleaned, explored, engineered new features and manipulated the data to make it usable for our use case.
* Used transfer learning with pre-trained model **DenseNet121** to train our model because it is smaller in size(33 MB) in comparison with other pre-trained models(making our web-app efficient) and also has optimal number of parameters(8.1M) and depth(242) which is perfect for our use case.
* Trained two dense models separately, regression and binary classifier, to predict age and gender respectively.
* Tuned our model hyper-parameters to get optimal results.



## **Data Cleaning & Pre-Processing:**
* We detect a some imbalances of age groups in our dataset upon checking distribution and value counts. We drop a fraction of them to normalize our age distribution.
![enter image description here](https://github.com/rishi5565/face-detection-with-age-and-gender-prediction/raw/main/EDA%20Images/1.png)
* Engineered new features from existing features in accordance with our project objective.
* Simplified complex features to better fit our model.
* Resized all images due to availability of limited resources and applied anti-aliasing to keep the images viable to detect key features.

## **Model Building:**
* We imported the pre-trained **DenseNet121** model to use in transfer learning because it is smaller in size(33 MB) in comparison with other pre-trained models(making our web-app efficient) and also has optimal number of parameters(8.1M) and depth(242) which is perfect for our use case.
* We made the last 15 layers trainable and set the parameters we want to use to train the model.
* We used adam optimizer in both our Age and Gender models with a low learning rate to prevent overfitting and other stability issues.
* We used mean squared error as our loss function as it is easy to interpret and we can simply round the output to get the predicted age.
* For our Gender model, we used sigmoid activation function on single neuron in final dense layer and "binary_crossentropy" as loss function as we have only 2 classes to predict from, Male and Female.
* For our Age model we finished training after 60 epochs and achieved a validation loss of MSE ~89 which was satisfactory. We also plotted the training and validation loss on a graph and concluded that there were no signs of overfitting or underfitting.
![enter image description here](https://github.com/rishi5565/face-detection-with-age-and-gender-prediction/raw/main/EDA%20Images/2.png)
* For our Gender model we finished training after 60 epochs and achieved a validation accuracy of ~90% which was satisfactory. We also plotted the training and validation loss and accuracy on graphs and concluded that there were no significant overfitting or underfitting.
![enter image description here](https://github.com/rishi5565/face-detection-with-age-and-gender-prediction/raw/main/EDA%20Images/3.png)
![enter image description here](https://github.com/rishi5565/face-detection-with-age-and-gender-prediction/raw/main/EDA%20Images/4.png)

**We saved both our models as h5 files and used them further in our Streamlit web-app.**

## **Conclusion**
We were able to build the front end of the app successfully using Streamlit. But there is definitely a lot of scope for improvement in performance. We could have achieved a much better result if we were not bound by Google Colab's limited resources and if we could use all the images in the dataset in their original shape of 200x200 to train our model. But for now we defined all the functions with all the necessary conditions and then proceeded to deploy the app on cloud for anyone to use.
**WEB-APP LINK:** [Face Detection with Age and Gender Prediction](https://share.streamlit.io/rishi5565/face-detection-with-age-and-gender-prediction/main/app.py)

Data Source: [Link](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### [Note]:  _**Please refer to the Project Notebook file for all the detailed in-depth information regarding this project.**_

Thank You,

Rishiraj Chowdhury ([rishiraj5565@gmail.com](mailto:rishiraj5565@gmail.com))
