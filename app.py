from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image
import streamlit as st
st.set_page_config(page_title="Face Detection with Age and Gender Prediction", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
# import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def get_img_and_predict(img):
  crop_faces = []
  age_list = []
  gender_list = []
  face_classifier = cv2.CascadeClassifier("models\\haarcascade_frontalface_default.xml")
  agemodel = load_model("models\\agemodel_densenet.h5")
  genmodel = load_model("models\\genmodel_densenet.h5")
  grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(grayimg, 1.3,5)

  for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi,(100,100), interpolation=cv2.INTER_AREA)
    crop_faces.append(roi)
    roi = roi.astype('float')/255
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    age = int(np.round(agemodel.predict(roi)))
    age_list.append(age)
    if genmodel.predict(roi) > 0.5:
      gender = "Female"
    else:
      gender = "Male"
    gender_list.append(gender)
  return crop_faces, age_list, gender_list


with st.sidebar:
    st.title("Face Detection with Age and Gender Prediction")
    st.write("Welcome to the web-app.")
    select_choice = st.selectbox(
        "Please select if you would like to use camera or upload an image.",
        ("Select one...", "Open Camera", "Upload Image"), index=0)

if select_choice == "Select one...":
    pass
elif select_choice == "Open Camera":
    pic = st.camera_input("Take a picture!")
    if pic:
        clicked = st.button("Click to detect faces and get prediction")
        if clicked == True:
            with st.spinner("Detecting and analyzing faces...."):
                pic = Image.open(pic)
                pic = np.array(pic)
                faces, age, gender = get_img_and_predict(pic)
                if len(faces) == 0:
                    st.warning("No faces detected!")
                else:
                    st.success("Success!! "+str(len(faces))+ " faces detected and analyzed.")
                    cols = st.columns(len(faces))
                    for i in range(0, len(faces)):
                        with cols[i]:
                            st.image(faces[i])
                            st.write("Age: " + str(age[i]))
                            st.write("Gender: " + str(gender[i]))
else:
    pic = st.file_uploader("Upload an Image", type=["jpg", "png", "webp"])
    if pic:
        st.image(pic, "Your uploaded image")
        clicked = st.button("Click to detect faces and get prediction")
        if clicked == True:
            with st.spinner("Detecting and analyzing faces...."):
                pic = Image.open(pic)
                pic = np.array(pic)
                faces, age, gender = get_img_and_predict(pic)
                if len(faces) == 0:
                    st.warning("No faces detected!")
                else:
                    st.success("Success!! "+str(len(faces))+ " faces detected and analyzed.")
                    cols = st.columns(len(faces))
                    for i in range(0, len(faces)):
                        with cols[i]:
                            st.image(faces[i])
                            st.write("Age: " + str(age[i]))
                            st.write("Gender: " + str(gender[i]))