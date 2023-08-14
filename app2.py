import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from io import *
import glob
import cv2
import tempfile
from datetime import datetime
import os
import os.path as osp

model = torch.hub.load('ultralytics/yolov5','custom',path='riding-m.pt')
def ImgPred(model) :
  image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
  col1, col2 = st.columns(2)
  col3, col4 = st.columns(2)
  if image_file is not None:
    img = Image.open(image_file)
    with col1:
      st.image(img, caption='Uploaded Image',use_column_width='always')
    with st.spinner(text="Predicting..."):
      # Load model
      st.header('Result :')
      pred = model(img)
      df = pred.pandas().xyxy[0]
      crop = df.values.tolist()
      crop = crop[0]
      imgCropped = img.crop(box=(crop[0:4]))
    with col2:
      st.image(imgCropped, caption='Model Prediction(s)',use_column_width='always')
    HM = torch.hub.load('ultralytics/yolov5','custom',path='helmet-m.pt')
    LM = torch.hub.load('ultralytics/yolov5','custom',path='LP-2.pt')
    with col3:
      pred1 = HM(imgCropped)
      pred1 = pred1.render()
      pred2 = LM(pred1)
      predren = pred2.render()
      st.image(predren, caption='Model Prediction(s)',use_column_width='always')
    with col4:

      df1 = pred2.pandas().xyxy[0]
      crop1 = df1.values.tolist()
      crop1 = crop1[0]
      imgCropped1 = img.crop(box=(crop1[0:4]))
      st.image(imgCropped1, caption='Model Prediction(s)',use_column_width='always')

def videoPred (model):
  uploaded_video = st.file_uploader( "Upload A Video", type=['mp4', 'mpeg', 'mov'])
  if uploaded_video is not None:
      tfile = tempfile.NamedTemporaryFile(delete=False)
      tfile.write(uploaded_video.read())
      video_name = uploaded_video.name
      fn , file_extension = osp.splitext(video_name)
      fn = ''.join(e for e in fn if e.isalnum()) + file_extension
      outputpath = osp.join('data/video_output', fn)
      os.makedirs('data/video_output', exist_ok=True)
      os.makedirs('data/video_frames', exist_ok=True)
      frames_dir = osp.join('data/video_frames',''.join(e for e in video_name if e.isalnum()))
      os.makedirs(frames_dir, exist_ok=True)
      frame_count = 0
      if uploaded_video:
            st.video(tfile.name)
            vid_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
              success, image = vid_cap.read()
              if success:
                frame_count += 1
                res = m(image)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                im = Image.fromarray(res_plotted[:,:,::-1])
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                im.save(osp.join(frames_dir, f'{frame_count}.jpg'))
              else :
                 vid_cap.release()
                 break
            os.system(
            f' ffmpeg -framerate 30 -i {frames_dir}/%d.jpg -c:v libx264 -pix_fmt yuv420p {outputpath}') 
            os.system(f'rm -rf {frames_dir}')
            output_video = open(outputpath, 'rb')
            output_video_bytes = output_video.read()
            st.video(output_video_bytes)      
def main() :
  st.title('Smort-CCTV')
  with st.sidebar:
    st.title("Option")
    option = st.selectbox('How would you like to be contacted?',('Image', 'Video'))
  if option == 'Video' :
    st.write('Using video upload option')
  else :
    st.write('Using image upload option')
  if option == 'Image':
    ImgPred(model) 
  else :
    videoPred(model)
if __name__ == "__main__":
  main()
