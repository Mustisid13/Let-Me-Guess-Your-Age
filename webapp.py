import streamlit as st
import numpy as np
from fastai.vision import load_learner,open_image,Image,pil2tensor
import PIL.Image
import io
import os
import PIL
import torch, torchvision
import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
# face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
@st.cache(suppress_st_warning=True)
def create_learner():
  return load_learner('/content/drive/My Drive/','export.pkl')

#detectron2
#==============================================================
@st.cache(suppress_st_warning=True)
def load_detectron():
  cfg = get_cfg()
  cfg.merge_from_file(
    model_zoo.get_config_file(
      "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
  )
  cfg.DATALOADER.NUM_WORKERS = 1
  cfg.SOLVER.IMS_PER_BATCH = 4
  cfg.SOLVER.BASE_LR = 0.001
  cfg.SOLVER.WARMUP_ITERS = 1000
  cfg.SOLVER.MAX_ITER = 1500
  cfg.SOLVER.STEPS = (1000, 1500)
  cfg.SOLVER.GAMMA = 0.05
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.TEST.EVAL_PERIOD = 50
  cfg.MODEL.WEIGHTS = '/content/drive/My Drive/model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
  predictor = DefaultPredictor(cfg)
  return predictor
#===============================================================
def prediction(f_buff):

  learner = create_learner()
  img = PIL.Image.open(f_buff)
  st.sidebar.image(img.resize((300,300)),caption="Uploaded Image")
  st.write('Detecting faces...')
  pix = np.asarray(img)
  output = pred(pix)
  boxes = output['instances'].get_fields()['pred_boxes'].tensor.to('cpu').numpy()
  flag=1
  for (x1,y1,x2,y2) in boxes:
    face_img=pix[int(y1):int(y2),int(x1):int(x2)]
        
    img_t = pil2tensor(face_img, np.float32)
    img_t.div_(255.0)
    image = Image(img_t)
    label = learner.predict(image)[0]
    st.write(f'{len(boxes)} Face Detected')
    
    #uncomment for 
    # v = Visualizer(pix[:,:,::-1],
    #         scale=0.5,
    #         instance_mode=ColorMode.IMAGE_BW)
    # out = v.draw_instance_predictions(output['instances'].to('cpu'))
    # st.image(out.get_image()[:,:,::-1],caption="Detectron2 visualization")
    
    st.image(face_img,caption="faces detected in image")
    st.write(f'''
    ## You look {label} years old.''')
  st.write(f"Am I correct??")
  if st.button('yes'):

    st.write('''
    ## Thank you for your feedback
    ''')
  if st.button('No'):
    st.write('''
    ## Oops! My Bad..
    ''')
  flag=0
  if flag==1:
    st.write('face not detected')
    
st.title("Let Me Guess Your Age!")

st.set_option('deprecation.showfileUploaderEncoding', False)
file_buffer = st.sidebar.file_uploader("Please Upload an Image", type=["jpg","png","jpeg"])
pred = load_detectron()
if file_buffer != None:
  prediction(file_buffer)

  
