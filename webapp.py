import streamlit as st
import numpy as np
from fastai.vision import load_learner,open_image,Image,pil2tensor
import PIL.Image
import io
import os
import cv2
import torch, torchvision
import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
# face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
learner = load_learner('','export.pkl')

#detectron2
#==============================================================

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
cfg.MODEL.WEIGHTS = 'model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
#===============================================================
def get_opencv_img_from_buffer(buffer, flags):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)


def prediction(f_buff):
    img = get_opencv_img_from_buffer(f_buff,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st.image(cv2.resize(img,(400,400)),caption="Uploaded Image")
    st.write('Detecting faces...')
    output = predictor(img)
    boxes = output['instances'].get_fields()['pred_boxes'].tensor.to('cpu').numpy()
    flag=1
    for (x1,y1,x2,y2) in boxes:

        face_img=img[int(y1):int(y2),int(x1):int(x2)]
        
        img_t = pil2tensor(face_img, np.float32)
        img_t.div_(255.0)
        image = Image(img_t)
        label = learner.predict(image)[0]
        st.write('Face Detected')
        '''
	#uncomment if you wish to see result of detectron2
        v = Visualizer(img[:,:,::-1],
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(output['instances'].to('cpu'))
        st.image(cv2.resize(out.get_image()[:,:,::-1],(300,300)),caption="Detectron2 visualization")
        '''
        st.image(cv2.resize(face_img,(224,224)),caption="faces detected in image")
        st.write(f"Prediction: {label}")
        flag=0
    # image = open_image(f_buff)
    if flag==1:
        st.write('face not detected')
    
    
        
    # defaults.device = torch.device('cpu')
    # pred_class= learner.predict(image)[0]
    
st.title("Let Me Guess Your Age!")

st.set_option('deprecation.showfileUploaderEncoding', False)
file_buffer = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
if file_buffer != None:
    prediction(file_buffer)
