import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import zipfile
import glob
import os


st.title('Image Decomposer')

# fastapi endpoint
url = 'http://fastapi:8000'
endpoint = '/segmentation'

st.write('''This application decomposes the image into multiple images of
objects in the original image. Implemented using Mask R-CNN model of
detectron2 library. The app can be used to get rid of the background, to
separate specific objects from the rest of the image, or/and image analysis.''') # description and instructions

image = st.file_uploader('insert image')  # image upload widget


def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r


if st.button('Get segmentation map'):
    r = process(image, url+endpoint)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    image_list = []
    for filename in glob.glob('*.png'):
        im=Image.open(filename)
        image_list.append(im)
    st.image(image_list, width=300) # output dyptich
    for filename in glob.glob('*.png'):
        os.remove(filename)
