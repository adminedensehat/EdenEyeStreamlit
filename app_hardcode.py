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

    if option == 'Brain (nifti)':
        st.subheader('Upload the MRI scan of the brain')
        uploaded_file = st.file_uploader(' ',accept_multiple_files = False)

        if uploaded_file is not None:
            file_path = os.path.join("BrainVolumes", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize model
            model = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=4,
                out_channels=3,
                dropout_prob=0.2,
            ).to(device)
            
            model.load_state_dict(torch.load(os.path.join(paths_dir, "brain_metric_model.pth"), map_location=torch.device('cpu')))
            model.eval()
            
            # Sort volumes
            path_test_volumes = sorted(glob(os.path.join(brain_vol_dir, "*")))
            test_files = [{"image": image_name} for image_name in path_test_volumes]
            
            # Transforms
            test_org_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(
                        keys=["image"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear"),
                    ),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )
            
            # Initialize MONAI dataset and loader
            test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
            test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=8)
            

            # Post transforms
            post_transforms = Compose(
                [
                    Invertd(
                        keys="pred",
                        transform=test_org_transforms,
                        orig_keys="image",
                        meta_keys="pred_meta_dict",
                        orig_meta_keys="image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cpu",
                    ),
                    Activationsd(keys="pred", sigmoid=True),
                    AsDiscreted(keys="pred", threshold=0.5),
                ]
            )
            
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            
            VAL_AMP = True
            # define inference method
            def inference(input):
                def _compute(input):
                    return sliding_window_inference(
                        inputs=input,
                        roi_size=(240, 240, 160),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                    )

                if VAL_AMP:
                    with torch.cuda.amp.autocast():
                        return _compute(input)
                else:
                    return _compute(input)
                    
            # Load the model state
            model.load_state_dict(torch.load(os.path.join(paths_dir, "brain_metric_model.pth"), map_location=torch.device('cpu')))
            model.eval()
            
            # Perform inference and visualization
            with torch.no_grad():
                # Select one image to evaluate and visualize the model output
                test_input = test_org_ds[0]["image"].unsqueeze(0).to(device)
                roi_size = (128, 128, 64)
                sw_batch_size = 4
                test_output = inference(test_input)

                # Number of slices to display (can adjust this as needed)
                num_slices = test_input.shape[4] # Assuming the slice dimension is the 4th one

                # Slider for selecting slice (in Streamlit)
                slice_idx = st.slider('Select a slice', 0, num_slices - 1, 0)

                # Visualize the selected slice
                fig, axes = plt.subplots(1, 4, figsize=(24, 6))
                for i in range(4):
                    axes[i].imshow(test_org_ds[0]["image"][i, :, :, slice_idx].detach().cpu(), cmap="gray")
                    axes[i].set_title(f"image channel {i} - Slice {slice_idx}")
                st.pyplot(fig)

                fig_output, axes_output = plt.subplots(1, 3, figsize=(18, 6))
                for i in range(3):
                    axes_output[i].imshow(test_output[0, i, :, :, slice_idx].detach().cpu())
                    axes_output[i].set_title(f"output channel {i} - Slice {slice_idx}")
                st.pyplot(fig_output)

            os.remove(file_path)    

        else:
            st.write("Make sure you file is in NIfTI Format.")     


    elif option == 'Lung (nifti)':
        # Drag and drop DICOM files or a zipped folder containing them
        uploaded_files = st.file_uploader("Upload DICOM files or a zipped folder containing DICOM files:", 
                                          accept_multiple_files=True)

        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Handle the uploaded files
                for uploaded_file in uploaded_files:
                    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Check if a zipped folder was uploaded and unzip if necessary
                if len(uploaded_files) == 1 and uploaded_files[0].name.endswith('.zip'):
                    with zipfile.ZipFile(os.path.join(temp_dir, uploaded_files[0].name), 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                # Set up output directory
                output_dir = os.path.join(temp_dir, 'nifti_output')
                os.makedirs(output_dir, exist_ok=True)

                # Perform the conversion
                try:
                    converter = DicomToNiftiConverter(min_images_per_series=1)
                    converter.convert(temp_dir, output_dir)

                    # Compress NIfTI files to .nii.gz format
                    for file in os.listdir(output_dir):
                        if file.endswith('.nii'):
                            with open(os.path.join(output_dir, file), 'rb') as f_in:
                                with gzip.open(os.path.join(output_dir, file + '.gz'), 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                    # Create a download button for the compressed NIfTI files
                    for file in os.listdir(output_dir):
                        if file.endswith('.nii.gz'):
                            with open(os.path.join(output_dir, file), 'rb') as f:
                                st.download_button('Download NIfTI File', f, file_name=file)

                    st.success('Conversion successful!')

                except Exception as e:
                    st.error(f'An error occurred: {e}')
        else:
            st.warning('Please upload DICOM files or a zipped folder.')


        # Segmentation task
        st.subheader('Upload the CT scans of the lung')
        uploaded_nifti = st.file_uploader(' ', accept_multiple_files=False)

        if uploaded_nifti is not None:
            # Upload file to LungVolumes folder
            st.write("NIfTI Uploaded Successfully")
            file_path = os.path.join(lung_vol_dir, uploaded_nifti.name)
            with open(file_path, "wb") as f: 
                f.write(uploaded_nifti.getbuffer()) 

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(device)

            model.load_state_dict(torch.load(
                os.path.join(paths_dir, "lung2_metric_model.pth"), map_location=torch.device('cpu')))
            model.eval()

            path_test_volumes = sorted(glob(os.path.join(lung_vol_dir, "*")))
            test_files = [{"image": image_name} for image_name in path_test_volumes]

            test_org_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["image"], source_key="image"),
                    RandAffined(keys=['image'], prob=0.5, translate_range=10),
                    RandRotated(keys=['image'], prob=0.5, range_x=10.0),
                    RandGaussianNoised(keys='image', prob=0.5),
                    Resized(keys=["image"], spatial_size=(128, 128, 64), mode='trilinear'),
                ]
            )

            test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
            test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)

            with torch.no_grad():
                for i, test_data in enumerate(test_org_loader):
                    # if i >= 100:  # Limit to first 50 images
                    #     break

                    roi_size = (512, 512, 512)
                    sw_batch_size = 4
                    test_outputs = sliding_window_inference(test_data["image"].to(device), roi_size, sw_batch_size, model)

                    # Load the NIfTI file to determine the number of slices
                    loader = LoadImage(image_only=True)
                    nifti_data = loader(file_path)
                    num_slices = nifti_data.shape[-1]  # Assuming the slices are along the last dimension

                    for j in range(num_slices): 
                        if j < test_data["image"].shape[4]:
                            fig = plt.figure("check", (12, 4))
                            plt.subplot(1, 2, 1)
                            plt.title(f"image {j}")
                            plt.imshow(test_data["image"][0, 0, :, :, j], cmap="gray")
                            plt.subplot(1, 2, 2)
                            plt.title(f"output {j}")
                            plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, j] == 1)
                            st.pyplot(fig)

            os.remove(file_path)                    

        else:
            st.write("Make sure you file is in NIfTI Format.")                   

    # elif option == 'Kidney (dicom)':
    #     st.subheader('Upload the CT scans of the kidney')
    #     uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
        
    #     if uploaded_files:
    #         for uploaded_file in uploaded_files:
    #             # Process each uploaded file
    #             st.write(f"Processing {uploaded_file.name}...")

    #             # Save each uploaded file to the ProstateVolumes folder
    #             file_path = os.path.join(kidney_vol_dir, uploaded_file.name)
    #             with open(file_path, "wb") as f: 
    #                 f.write(uploaded_file.getbuffer()) 


    #         st.write("All DICOM files uploaded and processed successfully.")
            
    #         # Initialize device
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    #         # Initialize model
    #         model = UNet(
    #             spatial_dims=3,
    #             in_channels=1,
    #             out_channels=2,
    #             channels=(32, 64, 128, 256, 512),
    #             strides=(2, 2, 2, 2),
    #             num_res_units=2,
    #             norm=Norm.BATCH,
    #         ).to(device)
            
    #         model.load_state_dict(torch.load(
    #             os.path.join(paths_dir, "kidney2_metric_model.pth"), map_location=torch.device('cpu')))
    #         model.eval()
            
    #         # Sort volumes
    #         path_test_volumes = sorted(glob(os.path.join(kidney_vol_dir, "*")))
    #         test_files = [{"image": kidney_vol_dir} for image_name in path_test_volumes]
            
    #         # Transforms
    #         test_org_transforms = Compose(
    #             [
    #                 LoadImaged(keys=["image"]),
    #                 EnsureChannelFirstd(keys=["image"]),
    #                 Spacingd(keys=["image"], pixdim=(1.84, 1.84, 2.36), mode="bilinear"),
    #                 NormalizeIntensityd(keys="image", subtrahend=103, divisor=73.3),
    #                 Resized(keys=["image"], spatial_size=(128, 128, 64), mode='trilinear'),
    #             ]
    #         )
            
    #         # Initialize MONAI dataset and loader
    #         test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
    #         test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)
            

    #         post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.3)])
            
    #         # Load the model state
    #         model.load_state_dict(torch.load(os.path.join(paths_dir, "kidney2_metric_model.pth"), map_location=torch.device('cpu')))
    #         model.eval()
            
    #         # Perform inference and visualization
    #         with torch.no_grad():
    #             for i, test_files in enumerate(test_org_loader):
    #                 if i >= 50:  # Stop after processing the first 50 images
    #                     break

    #                 roi_size = (128, 128, 128)
    #                 sw_batch_size = 4
    #                 test_outputs = sliding_window_inference(test_files["image"].to(device), roi_size, sw_batch_size, model)
    #                 # for i in range(50):
    #                 # Plot the slice [:, :, 80]
    #                 fig = plt.figure("check", (12, 4))
    #                 plt.subplot(1, 2, 1)
    #                 plt.title(f"image {i}")
    #                 plt.imshow(test_files["image"][0, 0, :, :, i], cmap="gray")
    #                 plt.subplot(1, 2, 2)
    #                 plt.title(f"output {i}")
    #                 plt.imshow(post_pred(test_outputs[0]).detach().cpu()[0, :, :, i])
    #                 # plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, i])
    #                 st.pyplot(fig)

    #         os.remove(file_path)                    
            
    #     else:
    #         st.write("Make sure you file is in Dicom Format.")

    elif option == 'Colon (nifti)':
        # Drag and drop DICOM files or a zipped folder containing them
        uploaded_files = st.file_uploader("Upload DICOM files or a zipped folder containing DICOM files:", 
                                          accept_multiple_files=True)

        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Handle the uploaded files
                for uploaded_file in uploaded_files:
                    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Check if a zipped folder was uploaded and unzip if necessary
                if len(uploaded_files) == 1 and uploaded_files[0].name.endswith('.zip'):
                    with zipfile.ZipFile(os.path.join(temp_dir, uploaded_files[0].name), 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                # Set up output directory
                output_dir = os.path.join(temp_dir, 'nifti_output')
                os.makedirs(output_dir, exist_ok=True)

                # Perform the conversion
                try:
                    converter = DicomToNiftiConverter(min_images_per_series=1)
                    converter.convert(temp_dir, output_dir)

                    # Compress NIfTI files to .nii.gz format
                    for file in os.listdir(output_dir):
                        if file.endswith('.nii'):
                            with open(os.path.join(output_dir, file), 'rb') as f_in:
                                with gzip.open(os.path.join(output_dir, file + '.gz'), 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                    # Create a download button for the compressed NIfTI files
                    for file in os.listdir(output_dir):
                        if file.endswith('.nii.gz'):
                            with open(os.path.join(output_dir, file), 'rb') as f:
                                st.download_button('Download NIfTI File', f, file_name=file)

                    st.success('Conversion successful!')

                except Exception as e:
                    st.error(f'An error occurred: {e}')
        else:
            st.warning('Please upload DICOM files or a zipped folder.')

        # Segmentation task
        st.subheader('Upload the CT scans of the colon')
        uploaded_nifti = st.file_uploader(' ', accept_multiple_files=False)

        if uploaded_nifti is not None:
            # Upload file to LiverVolumes folder
            st.write("NIfTI Uploaded Successfully")
            file_path = os.path.join(colon_vol_dir, uploaded_nifti.name)
            with open(file_path, "wb") as f: 
                f.write(uploaded_nifti.getbuffer()) 

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model=SegResNet(
                spatial_dims=3,
                init_filters=16,
                in_channels=1,
                out_channels=2,
                dropout_prob=0.2,
                act='RELU',
                num_groups=8,
                blocks_down=(1,2,2,4)
            ).to(device)

            model.load_state_dict(torch.load(
                os.path.join(paths_dir, "colon2_metric_model.pth"), map_location=torch.device('cpu')))
            model.eval()

            path_test_volumes = sorted(glob(os.path.join(colon_vol_dir, "*")))
            test_files = [{"image": image_name} for image_name in path_test_volumes]

            test_org_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["image"], source_key="image"),
                    Resized(keys=["image"], spatial_size=(128, 128, 64), mode='trilinear'),
                    ToTensord(keys=["image"]),
                ]
            )

            test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
            test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)

            with torch.no_grad():
                for i, test_data in enumerate(test_org_loader):
                    # if i >= 100:  # Limit to first 50 images
                    #     break

                    roi_size = (128, 128, 64)
                    sw_batch_size = 4
                    test_outputs = sliding_window_inference(test_data["image"].to(device), roi_size, sw_batch_size, model)

                    # Load the NIfTI file to determine the number of slices
                    loader = LoadImage(image_only=True)
                    nifti_data = loader(file_path)
                    num_slices = nifti_data.shape[-1]  # Assuming the slices are along the last dimension

                    for j in range(num_slices): 
                        if j < test_data["image"].shape[4]:
                            fig = plt.figure("check", (12, 4))
                            plt.subplot(1, 2, 1)
                            plt.title(f"image {j}")
                            plt.imshow(test_data["image"][0, 0, :, :, j], cmap="gray")
                            plt.subplot(1, 2, 2)
                            plt.title(f"output {j}")
                            plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, j] == 1)
                            st.pyplot(fig)

            os.remove(file_path)                    

        else:
            st.write("Make sure you file is in NIfTI Format.")  



    elif option == 'Prostate (dicom)':
        st.subheader('Upload the MRI scans of the prostate')
        uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Process each uploaded file
                st.write(f"Processing {uploaded_file.name}...")

                # Save each uploaded file to the ProstateVolumes folder
                file_path = os.path.join(prostate_vol_dir, uploaded_file.name)
                with open(file_path, "wb") as f: 
                    f.write(uploaded_file.getbuffer()) 


            st.write("All DICOM files uploaded and processed successfully.")

            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(device)

            model.load_state_dict(torch.load(
                os.path.join(paths_dir, "prostate_metric_model.pth"), map_location=torch.device('cpu')))
            model.eval()

            path_test_volumes = sorted(glob(os.path.join(prostate_vol_dir, "*")))
            test_files = [{"image": prostate_vol_dir} for image_name in path_test_volumes]

            test_org_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    ResampleToMatchd(keys="image", key_dst="image"),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 0.5), mode="bilinear"),  
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )

            test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
            test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)

            post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])



            with torch.no_grad():
                for i, test_file in enumerate(test_org_loader):
                    if i >= 50:
                        break

                    roi_size = (96, 96, 64)
                    sw_batch_size = 4
                    test_output = sliding_window_inference(test_file["image"].to(device), roi_size, sw_batch_size, model)

                    # for i in range(50):
                    fig = plt.figure("check", (12, 4))
                    plt.subplot(1, 2, 1)
                    plt.title(f"image {i}")
                    plt.imshow(test_file["image"][0, 0, :, :, i], cmap="gray")
                    plt.subplot(1, 2, 2)
                    plt.title(f"output {i}")
                    plt.imshow(post_pred(test_output[0]).detach().cpu()[0, :, :, i])
                    # plt.imshow(torch.argmax(test_output, dim=1).detach().cpu()[0, :, :, i])
                    st.pyplot(fig)

            # Code to delete files in prostate_vol_dir
            for filename in os.listdir(prostate_vol_dir):
                file_path = os.path.join(prostate_vol_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        else:
            st.write("Please upload DICOM files.")
            
            
    st.markdown("***")

    authenticator.logout('Logout', 'main')
elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

