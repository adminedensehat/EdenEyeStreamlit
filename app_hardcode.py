import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
#from preprocessing_images import *
import streamlit_authenticator as stauth

from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    # AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Invertd,
    AsDiscreted,
    SaveImaged,
    ScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    Activationsd,
)
from monai.networks.nets import UNet, SwinUNETR, SegResNet, SegResNetVAE
import glob
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from monai.inferers import sliding_window_inference
import shutil
from tqdm import tqdm
import dicom2nifti
import nibabel as nib
from monai.utils import first
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
st.markdown("<h1 style='text-align: center; color: white;'>Tumor Segmentation --</h1>", unsafe_allow_html=True)

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

