import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
#from preprocessing_images import *
import streamlit_authenticator as stauth
import glob
import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import shutil
import os
import tempfile
import pathlib
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate


####################### SETUP ##################################
# Directories
directory = os.getcwd()
paths_dir = os.path.join(directory, "Paths")
root_dir = tempfile.mkdtemp() if directory is None else directory

# For uploading
brain_vol_dir = os.path.join(directory, "BrainVolumes")
lung_vol_dir = os.path.join(directory, "LungVolumes")
kidney_vol_dir = os.path.join(directory, "KidneyVolumes")
prostate_vol_dir = os.path.join(directory, "ProstateVolumes")

######################## STREAMLIT ##########################

image = Image.open('EdenEye.png')
st.image(image)
st.markdown("<h1 style='text-align: center; color: white;'>Tuberculosis & Pneumonia Detector</h1>", unsafe_allow_html=True)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    st.write(f'Welcome *{st.session_state["name"]}*')

    option = st.radio('',('Brain (nifti)', 'Lung (nifti)', 'colon (nifti)', 'Prostate (dicom)'))
    st.write('You selected:', option)
    temp_dir = tempfile.TemporaryDirectory() # to save uploaded nifti

    
    st.markdown("***")

    authenticator.logout('Logout', 'main')
elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

