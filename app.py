import streamlit as st
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import cv2
import os
import tempfile

model = YOLO('my_modeln.pt')

def ImgPre(m) :
  image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
  if image_file is not None:
      img = Image.open(image_file)
      col1, col2 = st.columns(2)
      with col1 :
        st.image(img ,caption='Uploaded Image')

      with st.spinner(text="Predicting..."):
        # Load model
        pred = m(img)
        boxes = pred[0].boxes
        res_plotted = pred[0].plot()[:, :, ::-1]
        
      with col2 :
        st.image(res_plotted, caption='Detected Image',
            use_column_width=True,)


def videoPre (m):
  uploaded_video = st.file_uploader( "Upload A Video", type=['mp4', 'mpeg', 'mov'])
  if uploaded_video is not None:
      tfile = tempfile.NamedTemporaryFile(delete=False)
      tfile.write(uploaded_video.read())
      if uploaded_video:
            st.video(tfile.name)
            vid_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
              success, image = vid_cap.read()
              if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = m(image)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
              else :
                 vid_cap.release()
                 break
                  


def main() :

  st.title('Deployment Ai builder')

  with st.sidebar:
    st.title("Option")
    option = st.selectbox('How would you like to be contacted?',('Image', 'Video'))

  if option == 'Video' :
    st.write('Using video upload option')
  else :
    st.write('Using image upload option')

  if option == 'Image':
    ImgPre(model) 
  else :
    videoPre(model)

if __name__ == '__main__':
    main()
