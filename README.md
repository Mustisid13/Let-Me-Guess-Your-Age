# Let-Me-Guess-Your-Age-
This project detect age of person in the photo.

I have used Streamlit to build a webapp for my model. Streamlit is a python package which helps in building webapps easily and with a few lines of code. you can host your webapp on on heroku

Also used Detectron2 for detecting all the faces in the image.

Used google colab for training detectron2 for face detection.

Dataset used for training model for age detection: https://www.kaggle.com/frabbisw/facial-age

Dataset used for training detectron2 for face detection: https://www.kaggle.com/dataturks/face-detection-in-images

My kaggle Notebook (age detection model):https://www.kaggle.com/mustisid/facial-age-detection-fastai

download detectron2 pre-trained model: https://drive.google.com/file/d/1-3piwYHz1I5kMuSUkr3Dj1cMciOr12ZJ/view?usp=sharing

References for detectron2: 
 - https://www.curiousily.com/posts/face-detection-on-custom-dataset-with-detectron2-in-python/
 - https://detectron2.readthedocs.io/tutorials/getting_started.html 
 - https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=b-i4hmGYk1dL

References for fastai:
- https://course.fast.ai/terminal_tutorial.html

Note: If you want to run above code on cpu use requirementsCPU.txt to install dependencies
