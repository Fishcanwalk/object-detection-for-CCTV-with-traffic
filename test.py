import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
import os.path as osp

st.title('Deployment Ai builder')
st.title('object-detection-for-CCTV-with-traffic')
with st.sidebar:
  st.title("Option")
  option = st.selectbox('How would you like to be contacted?',('Image', 'Video'))
  st.title("Select_Model")
  option = st.selectbox('How would you like to be contacted?',('Model-v5', 'Model-v8')) 
    
if option == 'Video' :
  st.write('Using video upload option')
else :
  st.write('Using image upload option')
    
if option == 'Image':
  if Select_Model == 'Model-v5':
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='mymodelv5.pt', force_reload=True)
  else:
    model = YOLO('my_modeln.pt')
  ImgPre(model) 
else :
  if Select_Model == 'Model-v5':
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='mymodelv5.pt', force_reload=True)
  else:
    model = YOLO('my_modeln.pt')
  videoPre(model)

def ImgPre(m) :
  image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
  if image_file is not None:
      img = Image.open(image_file)
      st.image(img ,caption='Uploaded Image')
      with st.spinner(text="Predicting..."):
        # Load model
        pred = m(img,conf = 0.2)
        boxes = pred[0].boxes
        res_plotted = pred[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image')


def videoPre (m):
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
                result_tensor = res[0].boxestorch.hub.load('ultralytics/yolov5', 'custom',
                           path=CFG_MODEL_PATH, force_reload=True, device=device)
                res_plotted = res[0].plot()
                im = Image.fromarray(res_plotted[:,:,::-1])
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )Error: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
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
        

    

