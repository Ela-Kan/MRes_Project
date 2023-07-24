"""
Main class to process 3D-FLAIR images within the pipeline for registration and subtraction of 3D FLAIR sequences for 
Amyloid Related Abnormality (ARIA) detection. T1 processing and conversion should be performed prior to using the methods in this object.

Author: Ela Kanani

Required dependencies: 
    * Chris Rorden's dcm2niiX version v1.0.20171215 (OpenJPEG build) GCC7.3.0 (64-bit Linux)
    * nipype v1.8.6 Gorgolewski et. al 2011
    *  *IGNORE*brainextractor v0.2.2 (repackaged FSL BET) https://pypi.org/project/brainextractor/*IGNORE*

Optional dependencies:
    * 'pigz': allows for faster compression of images

"""
import numpy as np
import os
import pandas as pd
import shutil
import nipype.interfaces.fsl as fsl
import matplotlib.pyplot as plt
from pathlib import Path
import Registration as rg # import my own registration module

# for intensity normalisation
import nibabel as nib
from intensity_normalization.typing import Modality
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.plot.histogram import HistogramPlotter, plot_histogram

# import the nibabel library so we can read in a nifti image
#import nibabel as nib
# import the BrainExtractor class
#from brainextractor import BrainExtractor


class Process3DFLAIR():

    def __init__(self, all_subject_info, subject_id, total_num_time_points, time_points_to_consider, registration_method, inter_subject = False) -> None:
        """
        Initialises instance of the preprocess3DFLAIR object.

        Parameters
        ----------

        all_subject_info : pd.Dataframe
            dataframe including information on all subjects. contains a column of subject ids and corresponding total number of timeframes. See 
            documentation for further information

        subject_id : str
            string including current subject id. e.g. 'B-RAP_0027'
        
        total_num_time_points : int
            number of scans in the temporal series

        time_points_to_consider : list (of ints)
            the time points to consider in analysis. E.g. [1,2] to consider first and second time point. Ensure
            that these are in chronological order.
    
        registration_method : str
           the type of registration to run on the images.
           Options:
                rigidfsl: rigid registration using FSL's FLIRT

        inter_subject: bool
            whether the normalisation should be performed inter or intra subject. default is False. If True,
            then a list of all subject IDs is required.


        Returns
        -------
        None

        """
        
        # Define the global variables to be used throughout the implementations detailed.
        self.subject_id = subject_id # the subject id for the 3D FLAIR to be preprocessed
        self.all_subject_info = all_subject_info # pandas dataframe of all subject information
        self.total_num_time_points = total_num_time_points # the number of temporal scans for the given subject
        self.time_points_to_consider = time_points_to_consider # the specific scans to analyse
        self.registration_method = registration_method #string including the desired registration method to use
        self.inter_subject = inter_subject # boolean indicating whether inter- or intra- subject normalisation is desired

        # create a folder tree with the appropriate file paths for use throughout this method, as according to accompanying doc
        self.subject_directory_root = '/home/ela/Documents/B-RAPIDD/' + self.subject_id +'/3D-FLAIR/'
        self.subject_dicom_directory = self.subject_directory_root + 'original_dicom/' # this folder should already exist with the downloaded data
        self.subject_T1_directory = '/home/ela/Documents/B-RAPIDD/' + self.subject_id + '/T1-MPRAGE/' # T1 preprocessing should be done before using this function
        self.registered_directory = self.subject_directory_root + 'registered_nifti/' #folder containing the registered temporal scans
        self.subject_bias_directory = self.subject_directory_root + 'bias_nifti/'
        self.subject_bias_fields_directory = self.subject_bias_directory  + 'fields/'
        os.makedirs(self.subject_bias_fields_directory, exist_ok=True)     

        # create folder for storing normalised images
        self.subject_normalised_directory = self.subject_directory_root + 'normalised_nifti/'
        os.makedirs(self.subject_normalised_directory, exist_ok=True)

        # create folder for FLIRT mat if they do not exist
        self.subject_FLIRT_mat_directory = self.registered_directory + 'FLIRT_mat/'
        os.makedirs(self.subject_FLIRT_mat_directory, exist_ok=True)

        # if the folders don't exist, create them
        os.makedirs(self.registered_directory , exist_ok=True)
        os.makedirs(self.subject_bias_directory, exist_ok=True)
        
        # create a folder for the converted NIFTI images and brain extractions for the 3D-FLAIR
        self.subject_nifti_directory = self.subject_directory_root + 'original_nifti/'
        os.makedirs(self.subject_nifti_directory, exist_ok=True)
        self.subject_brain_directory = self.subject_directory_root + 'brain_nifti/'
        os.makedirs(self.subject_brain_directory, exist_ok=True)
        self.subject_brain_mask_directory = self.subject_brain_directory + 'masks/'
        os.makedirs(self.subject_brain_mask_directory, exist_ok=True)

        # create variables for the T1 directories
        self.subject_T1_nifti_directory = self.subject_T1_directory + 'original_nifti/'
        self.subject_T1_brain_directory = self.subject_T1_directory + 'brain_nifti/'


        
        print("Analysing subject: " + self.subject_id)
        return None
    
    def findDICOMFolder(self, time_point):
        """
        Finds the file path of a child folder containing DICOM images only. Useful when there is a large 
        tree of folders, as often found in the B-RAPIDD set. 

        Parameters
        ----------
        time_point : int
            Specific time point to process (e.g., first scan would be 1)  
                
        Returns
        -------
        folder_path : str
            string containing the folder path for the folder containing all of the dicom images for a given
            subject

        """
        dicom_root_path = self.subject_dicom_directory + self.subject_id+ f'_{str(time_point).zfill(2)}_D1' 

        print(self.subject_id + f'_{str(time_point).zfill(2)}_D1')

        for dirpath, dirnames, filenames in os.walk(dicom_root_path):
            # Iterate through all folders and files in the root_path and its subdirectories
            for dirname in dirnames:
                folder_path = os.path.join(dirpath, dirname) # store the folder path
                dicom_files = [file for file in os.listdir(folder_path) if file.endswith('.dcm')] # if a dicom file is within the folder, return the path
                if dicom_files:
                    return folder_path      



    def renameNIFTIFiles(self):
        """
        dcm2niiX outputted NIFTI and corresponding json files do not follow the desired convention. This function
        takes the directory of a folder containing NIFTI files and their corresponding json files, and splits them
        into two folders. Next, each file is renamed to follow the same convention as the provided B-RAPIDD dicom scans.

        Parameters
        ----------
        None        
 
        Returns
        -------
        None

        """
    
        # need to edit directory input for os processing (must be in the form /home/ela/Documents/B-RAPIDD/subject_id/3D-FLAIR/original_nifti/subject_id_01_D1)
        #directory = '/home/ela'+ directory
        # store list of files in the current directory
        root_file_list = os.listdir(self.subject_nifti_directory)

        # move json files into a separate folder called json_info
        json_directory =  self.subject_nifti_directory + 'json_info'
        os.makedirs(json_directory, exist_ok=True) # create the json folder if it doesn't exist
        for file in root_file_list:
            if file.endswith('.json'): # if the file is a json file move it
                shutil.move(os.path.join(self.subject_nifti_directory, file), os.path.join(json_directory, file))

        # sort the json and nifti file names in their corresponding folders in ascending order
        nifti_file_list = os.listdir(self.subject_nifti_directory) # update nifti-only file list
        json_file_list = os.listdir(json_directory) # update nifti-only file list
        nifti_file_list.sort()
        json_file_list.sort()

        # count the number of files, each nifti has a corresponding json so this should be an equal number
        file_count = len(nifti_file_list)

        # rename the files in each folder to match the naming convention in the accompanying material
        for index, (nifti_file_list, json_file_list) in enumerate(zip(nifti_file_list, json_file_list), start=1):
            new_file_name = self.subject_id+f'_{str(index).zfill(2)}_D1' # create string of subject ID and padded temporal (e.g. 01) reference, e.g. B-RAP_0027_01_D1
            new_nifti_name = f"{new_file_name}.nii.gz"
            new_json_name = f"{new_file_name}.json"
            # rename the files
            os.rename(os.path.join(self.subject_nifti_directory, nifti_file_list), os.path.join(self.subject_nifti_directory, new_nifti_name))
            os.rename(os.path.join(json_directory, json_file_list), os.path.join(json_directory, new_json_name))

        return None

    
    def convertDICOMtoNIFTI(self):
        """
        Converts the current subject's DICOM to NIFTI following the file tree structure in folder. 
        Requries Chris Rorden's dcm2niiX.

        Parameters
        -------
        None
                
        Returns
        -------
        None

        """
        
        # initialse nifti file type
        nifti_file_format = self.subject_nifti_directory + self.subject_id+'_{}_D1.nii.gz'

        # iterate through all images in the desired number of files
        for i in range(self.total_num_time_points):
            # each dicom folder contains a sub-directory of variable folder names. Therefore, we need to find the whole path
            # for the i+1th time point
            current_dicom_loc = self.findDICOMFolder(i+1)
            
            # check if the converted nifti file already exists, if so continue to the next time point
            current_img_str = str(0)+str(i+1) # current time point string
            print(nifti_file_format.format(current_img_str))
            if os.path.isfile(nifti_file_format.format(current_img_str)):
                continue
            else: # if the time point needs to be converted
                dcm2niix_cmd = 'dcm2niix -o ' +  self.subject_nifti_directory + ' -z y -f %f ' + current_dicom_loc
                # call the dcm2niix method
                os.system(dcm2niix_cmd)

        # change the nifti file names to fit convention (to match the corresponding DICOM name)
        self.renameNIFTIFiles()

        # reformat the images into standard format
        for i in range(self.total_num_time_points):
            current_img_str = str(0)+str(i+1) # current time point string
            reorient = fsl.Reorient2Std() 
            reorient.inputs.in_file = nifti_file_format.format(current_img_str)
            reorient.inputs.out_file = nifti_file_format.format(current_img_str)
            res = reorient.run()

        print('Finished conversion and reorientation.')
        return None
    
    def extractBrainNIFTIold(self, useT1 = False):
        """
        Write Documentation 
        """
        # takes subject ID and extracts the brains at each time point using BET

        # initialise nifti and brain file format to iterate over
        nifti_file_format = self.subject_nifti_directory + self.subject_id+'_{}_D1.nii.gz'
        brain_file_format = self.subject_brain_directory + self.subject_id+'_{}_D1.nii.gz'
        T1_nifti_file_format = self.subject_T1_nifti_directory + self.subject_id+'_{}_D1.nii.gz'
        T1_brain_file_format = self.subject_T1_brain_directory + self.subject_id+'_{}_D1.nii.gz'
        T1_brain_masks_file_format = self.subject_T1_brain_directory + self.subject_id+'_{}_D1_mask.nii.gz'
        registered_T1_file_format = self.subject_T1_directory + 'T1_in_FLAIR_nifti/' + self.subject_id+'_{}_D1.nii.gz'


        # iterate through all images in the desired number of time points
        for i in range(len(self.time_points_to_consider)):
            current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
            current_brain_loc = brain_file_format.format(current_img_str)
            # check if the extracted brain exists for the current time point
            if os.path.exists(current_brain_loc):
                continue # move onto next time point if the brain is already extracted
            else: # if there isn't an extracted brain
                current_nifti_loc =  nifti_file_format.format(current_img_str) # current FLAIR image
                
                # if we are not using the T1 for brain extraction
                if useT1 == False: 
                    bet = fsl.BET() # initialise fsl brain extraction object
                    extracted_brain = bet.run(in_file = current_nifti_loc, out_file = current_brain_loc, frac=0.45)
                    print('Extracted 3D-FLAIR brain at time point: ' + str(i+1))
               
                # if we are using the T1 for brain extraction
                else: 
                    current_T1_loc = T1_nifti_file_format.format(current_img_str) # current T1 image
                    current_registered_T1_loc = registered_T1_file_format.format(current_img_str)
                    # Step 1: Register T1 into 3D FLAIR space using FLIRT 
                    # check if the T1 has already been registered into 3D-FLAIR space (i.e. if the file exists, if not then run)
                    if not os.path.isfile(current_registered_T1_loc):
                        T1_FLAIR_flt = fsl.FLIRT(
                                            cost_func = 'mutualinfo',
                                            dof = 6, # rigid body transformation
                                            interp = 'trilinear')
                        T1_FLAIR_flt.inputs.in_file = current_T1_loc
                        T1_FLAIR_flt.inputs.reference = current_nifti_loc
                        T1_FLAIR_flt.inputs.output_type = "NIFTI_GZ" # save as a NIFTI file
                        T1_FLAIR_flt.inputs.out_file = current_registered_T1_loc
                        #T1_FLAIR_flt.inputs.out_matrix_file = 'subject_to_template.mat'
                        T1_FLAIR_result= T1_FLAIR_flt.run() 
                        print('Registered T1 to 3D FLAIR at time point: '+ current_img_str)
                    
                    # Step 2: extract T1 brain and create a mask if it doesn't already exist
                    current_T1_brain = T1_brain_file_format.format(current_img_str) # current T1 brain extracted 
                    if not os.path.isfile(current_T1_brain): # if the current mask doesn't exist
                        # run BET pipeline
                        T1_bet = fsl.BET()
                        T1_bet.inputs.in_file = current_registered_T1_loc
                        T1_bet.inputs.frac = 0.4 # fractional intensity threshold
                        T1_bet.inputs.mask = True 
                        T1_bet.inputs.out_file = current_T1_brain
                        T1_bet_res = T1_bet.run() # run brain extraction with desired inputs
                        print('Extracted T1 brain at time point: ' + current_img_str)

                    # Step 3: apply T1 mask to 3D FLAIR
                    apply_mask_BET = fsl.ApplyMask()
                    apply_mask_BET.inputs.in_file = current_nifti_loc # input current 3D FLAIR
                    apply_mask_BET.inputs.mask_file = T1_brain_masks_file_format.format(current_img_str) 
                    apply_mask_BET.inputs.out_file = current_brain_loc
                    FLAIR_bet_res = apply_mask_BET.run()
                    print('Extracted 3D-FLAIR brain at time point: ' + current_img_str)
        return None
    
    def extractBrain(self):
        # TODO: use HD-BET method here
        
        # extract the brain here, if using a biased registration take the final time point as the template
        brain_template = self.time_points_to_consider[-1] # select last value of the time points to consider for the template
        # iterate through all of the time points except the last
        for i in range(len(self.time_points_to_consider)-1):
            self.applyBETMask(extracted_timepoint=brain_template, to_extract_timepoint=self.time_points_to_consider[i])

        return None 
    
    def applyBETMask(self, extracted_timepoint, to_extract_timepoint):
        """
        Applies a binary mask from an extracted brain (preferably from HD-BET in this pipeline), to
        a brain of choice. This allows for consistency in brain volumes. Initially registration is required
        between the extracted time point and the time to extract, which is done using the desired input.

        Parameters
        ----------
        extracted_timepoint : int
            Specific time point of the brain that has an extracted mask (e.g., first scan would be 1)  
        to_extract_timepoint : int
            Specific time point for brain we want to extract (e.g., second scan would be 2)  

                
        Returns
        -------
        None

        """
        
        # define the file path format layout
        nifti_file_format = self.subject_nifti_directory + self.subject_id+'_{}_D1.nii.gz'
        brain_file_mask_format = self.subject_brain_directory + 'masks/'+ self.subject_id+'_{}_D1.nii'  
        brain_file_format = self.subject_brain_directory + self.subject_id+'_{}_D1.nii.gz' 
        affine_file_path_format = self.registered_directory + '/FLIRT_mat/' + self.subject_id+'_{}_D1_flirt.mat'

        # specify file names using the above formatting
        extracted_brain = brain_file_format.format(str(extracted_timepoint).zfill(2))
        brain_to_extract = nifti_file_format.format(str(to_extract_timepoint).zfill(2)) 
        mask = brain_file_mask_format.format(str(extracted_timepoint).zfill(2))
        registered_whole_brain_format = self.registered_directory + self.subject_id+'_{}_D1.nii.gz'
        registered_whole_brain = registered_whole_brain_format.format(str(to_extract_timepoint).zfill(2))
        

        # need to register the two time points using desired registration method
        if self.registration_method == 'rigidfsl': # if we want rigid FLIRT 
            rigidRegister = rg.Registration(reference_path = extracted_brain, target_path = brain_to_extract, out_path = registered_whole_brain)
            rigidRegister.rigidFslFLIRT(cost_function = 'mutualinfo', interpolation = 'trilinear')
            print('Registered time point '+ str(to_extract_timepoint) + ' to ' + str(extracted_timepoint))

        if self.registration_method == 'affinefsl': # if we want affine FLIRT 
            affineRegister = rg.Registration(reference_path = extracted_brain, target_path = brain_to_extract, out_path = registered_whole_brain)
            affineRegister.affineFslFLIRT(cost_function = 'mutualinfo', interpolation = 'trilinear')
            print('Registered time point '+ str(to_extract_timepoint) + ' to ' + str(extracted_timepoint))

        if self.registration_method == 'nonlinearfsl': #non-linear fsl. NOTE: this requires affine matrix first (RERUN AFFINE)
            nonlinearRegister = rg.Registration(reference_path = extracted_brain, target_path = brain_to_extract, out_path = registered_whole_brain)
            affine_file_path = affine_file_path_format.format(str(to_extract_timepoint).zfill(2))
            nonlinearRegister.nonlinearFslFNIRT(affine_file_path)
            print('Registered time point '+ str(to_extract_timepoint) + ' to ' + str(extracted_timepoint))
        # extract the brain following registration
        apply_mask_BET = fsl.ApplyMask()
        apply_mask_BET.inputs.in_file = registered_whole_brain
        apply_mask_BET.inputs.mask_file = mask
        apply_mask_BET.inputs.out_file = brain_file_format.format(str(to_extract_timepoint).zfill(2))
        FLAIR_bet_res = apply_mask_BET.run()
        print('Extracted 3D-FLAIR brain at time point: ' + str(to_extract_timepoint))

        return None


    def correctBiasField(self, method = 'FSL'):

        # initialise brain file format to iterate over
        brain_file_format = self.subject_brain_directory + self.subject_id+'_{}_D1.nii.gz'
        
        bias_file_format = self.subject_bias_directory + self.subject_id +'_{}_D1.nii.gz'

        # correct the bias field of the images 
        if method == 'FSL':
            brain_nifti_filenames = [] # list of brain file namesPath(nifti_file_format.format(current_img_str)) for later use
          # iterate through all images in the desired number of time points
            for i in range(len(self.time_points_to_consider)):
                current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
                print('Correcting bias field at time point: ' + current_img_str)
                current_brain_loc = brain_file_format.format(current_img_str)
                current_bias_loc = bias_file_format.format(current_img_str)
                FLAIR_fast = fsl.FAST()
                FLAIR_fast.inputs.in_files = current_brain_loc
                FLAIR_fast.inputs.bias_iters = 5
                FLAIR_fast.inputs.img_type = 1 # T1 should be sufficient for FLAIR
                FLAIR_fast.inputs.number_classes = 5 # WM, GM, CSF, small lesions, large lesions (like in paper NOTE: add citation from zotero methods)
                #FLAIR_fast.inputs.out_basename = current_bias_loc, this is broken in nipype
                FLAIR_fast.inputs.output_biascorrected = True
                FLAIR_fast.inputs.output_biasfield = True
                FLAIR_fast_res = FLAIR_fast.run()
                print('Corrected bias field at time point: ' + current_img_str)
                brain_nifti_filenames.append(current_brain_loc)


            # Reorganise file tree
            source_folder = self.subject_brain_directory
            target_folder = self.subject_bias_directory
            bias_subfolder = self.subject_bias_fields_directory

            # Iterate over the files in the source folder
            for filename in os.listdir(source_folder):
                if filename.startswith(self.subject_id):
                    base_name = filename.rsplit(".", 2)[0]  # Remove the file extension
                    suffix = base_name.rsplit("_", 1)[1]  # Get the last part of the base name
                    # Check if the file is 'restore' or 'bias' and move it accordingly
                    if suffix == "restore":
                        target_path = os.path.join(target_folder, filename)
                        shutil.move(os.path.join(source_folder, filename), target_path)
                    elif suffix == "bias":
                        target_path = os.path.join(bias_subfolder, filename)
                        shutil.move(os.path.join(source_folder, filename), target_path)
                    # delete files that are unneccessary
                    elif suffix == "mixeltype":
                        os.remove(os.path.join(source_folder, filename))
                    elif suffix == "seg":
                        os.remove(os.path.join(source_folder, filename))
                    elif "pve" in base_name:
                        os.remove(os.path.join(source_folder, filename))
                    else:
                        continue  # skip files with other suffixes
                    
            return None

    def intensityNormalisation(self, useBiasCorrected = True):

        if self.inter_subject == False: # if we only want to normalise according to the current subject's data
            # save image paths into a list
            if useBiasCorrected == True:
                brain_file_format = self.subject_bias_directory + self.subject_id +'_{}_D1_restore.nii.gz'
            else:
                brain_file_format = self.subject_brain_directory + self.subject_id+'_{}_D1.nii.gz'
            image_paths = [] # initialise variable
            for i in range(len(self.time_points_to_consider)):
                current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
                image_paths.append(brain_file_format.format(current_img_str))

            # load in images for processing
            images = [nib.load(image_path).get_fdata() for image_path in image_paths]        

            # normalise images
            nyul_norm = NyulNormalize()
            nyul_norm.fit(images, modality = Modality.FLAIR)
            normalized = [nyul_norm(image) for image in images]
            nyul_norm.save_standard_histogram(self.subject_normalised_directory + "standard_histogram.npy")
        
            # show histogram of original and corrected images for validation
            hp = HistogramPlotter(title="Original Intensities")
            masks = None
            _ = hp(images, masks)
            plt.show()
            hp = HistogramPlotter(title="Nyul Normalised Intensities")
            masks = None
            _ = hp(normalized, masks)
            plt.show()

        if self.inter_subject == True: # consider all subjects/multiple subjects for normalisation training, requires bias field correction and registration on all. 
            bias_corrected_brain_format = '/home/ela/Documents/B-RAPIDD/{}/3D-FLAIR/bias_nifti/{}_{}_D1_restore.nii.gz' #1st and 2nd blank are subject IDS, third is timepoint
            brain_mask_format = '/home/ela/Documents/B-RAPIDD/{}/3D-FLAIR/brain_nifti/masks/{}_{}_D1.nii'
            image_paths = [] # initialise variables
            mask_paths = []
            image_subject_labels = []
            # save image and mask paths to a list
            for i in range(len(self.all_subject_info.Subject_ID[:])): # for all of the subjects in the dataframe, load in all of the brain extracted images
                current_subject_ID = self.all_subject_info.Subject_ID[i] # extract the current subject's ID
                for t in range(self.all_subject_info.Time_Points[i]): # for each time point in the current subject's data
                    # store image paths
                    current_time_str = str(0)+str(t+1) # current time point string
                    image_paths.append(bias_corrected_brain_format.format(current_subject_ID, current_subject_ID, current_time_str))
                    # store mask paths(this will be the same across all brains so repeatedly store the same path t times)
                    mask_paths.append(brain_mask_format.format(current_subject_ID, current_subject_ID, str(0) + str(self.all_subject_info.Time_Points[i]))) # select the final time point to consider as the mask
                    # save the corresponding subject labels as a key for the images/positioning of data for a given subject ID
                    image_subject_labels.append(current_subject_ID)

            # load in images for processing
            images = [nib.load(image_path).get_fdata() for image_path in image_paths]         
            # load in masks for processing
            masks = [nib.load(mask_path).get_fdata() for mask_path in mask_paths]     

            # normalise images 
            nyul_norm = NyulNormalize()
            nyul_norm.fit(images, masks, modality = Modality.FLAIR)
            normalized = [nyul_norm(image) for image in images]
            nyul_norm.save_standard_histogram(self.subject_normalised_directory + "standard_histogram.npy")   

            # show histogram of original and corrected images for validation
            hp = HistogramPlotter(title="Original Intensities")
            _ = hp(images, masks)
            plt.show()
            hp = HistogramPlotter(title="Nyul Normalised Intensities")
            _ = hp(normalized, masks)
            plt.show()       

            # only retain the desired subject's normalised data for future use
            current_subject_normalized = []
            for i, id in enumerate(image_subject_labels):
                # check if the current ID matches the subject we are considering
                if id == self.subject_id:
                    current_subject_normalized.append(normalized[i]) # save the normalised images for the current subject
            normalized = current_subject_normalized # remove images that we are not currently considering

        # save normalised images
        normalised_file_format = self.subject_normalised_directory + self.subject_id+'_{}_D1.nii.gz'
        for i in range(len(self.time_points_to_consider)):
            current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
            norm_nifti = nib.Nifti1Image(normalized[i], affine=np.eye(4))
            nib.save(norm_nifti, normalised_file_format.format(current_img_str))

        # reformat the images into standard format
        for i in range(len(self.time_points_to_consider)):
            current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
            reorient = fsl.Reorient2Std() 
            reorient.inputs.in_file = normalised_file_format.format(current_img_str)
            reorient.inputs.out_file = normalised_file_format.format(current_img_str)
            res = reorient.run()

    def subtractImages(self, in_image_num, image_to_subtract_num, out_file, threshold = True):
        # input = time point flag for images e.g. 1 or 2

        # load in the images
        normalised_file_format = self.subject_normalised_directory + '/nonlinear/' + self.subject_id+'_{}_D1.nii.gz'
        in_image_path = normalised_file_format.format(str(in_image_num).zfill(2))
        image_to_subtract_path = normalised_file_format.format(str(image_to_subtract_num).zfill(2))

        # load in images for processing and convert to numpy arrays
        in_image = np.array(nib.load(in_image_path).get_fdata())
        image_to_subtract = np.array(nib.load(image_to_subtract_path).get_fdata())

        # subtract images
        subtraction_img = in_image - image_to_subtract

        # if we want to create a threshold image
        if threshold == True:
            subtraction_img[subtraction_img < 0] = 0

        # save subtraction image
        sub_nifti = nib.Nifti1Image(subtraction_img, affine=np.eye(4))
        nib.save(sub_nifti, out_file)

        print('Subtracted point ' + str(image_to_subtract_num) + ' from ' + str(in_image_num))
        
        return None
    
    def runSubtraction(self, in_image_num, image_to_subtract_num, out_file):
        """ *REQUIRES IMAGES TO ALREADY BE IN NIFTI FORMAT*
        
        """

        # Extract brain at final time point in sequence using HD-BET

        # Step 1) Extract brain using HD-BET and mask
        self.extractBrain()
        # Step 2) Perform bias field correction
        self.correctBiasField(method = 'FSL')
        # Step 3) Perform intensity normalisation
        self.intensityNormalisation(useBiasCorrected = True)
        # Step 4: Compute subtraction map
        self.subtractImages(in_image_num, image_to_subtract_num, out_file)


        return None

    def calcVariance(self, out_file):
        # load in the images
        normalised_file_format = self.subject_normalised_directory + self.subject_id+'_{}_D1.nii.gz'
        image_paths = [] # initialise variable
        for i in range(len(self.time_points_to_consider)):
                    current_img_str = str(0)+str(self.time_points_to_consider[i]) # current time point string
                    image_paths.append(normalised_file_format.format(current_img_str))
        # load in images for processing
        images = [nib.load(image_path).get_fdata() for image_path in image_paths]

        # preallocate array of variances 
        variances = np.zeros(images[0].shape)

        # iterate through each 'voxel' in the images
        for i in range(variances.shape[0]):
                for j in range(variances.shape[1]):
                        for k in range(variances.shape[2]):
                                voxel_intensities = np.zeros((1, len(self.time_points_to_consider)))
                                # extract corresponding intensities 
                                for n in range(len(self.time_points_to_consider)):
                                    voxel_intensities[0,n] = images[n][i,j,k]
                                # calculate variance 
                                variances[i,j,k] = np.var(voxel_intensities)

        # save variance map
        var_nifti = nib.Nifti1Image(variances, affine=np.eye(4))
        nib.save(var_nifti, out_file)
        
        return None
    
    def runVariancePipeline(self, out_file):
        # Requires images to be in NIFTI already

        # Step 1) Extract brain using HD-BET and mask
        #self.extractBrain()
        # Step 2) Perform bias field correction
        #self.correctBiasField(method = 'FSL')
        # Step 3) Perform intensity normalisation
        self.intensityNormalisation(useBiasCorrected = True)
        # Step 4: Compute variance map
        self.calcVariance(out_file)

        print('Completed variance map')
       
        return None

if __name__ == "__main__":

    # open the subject info table and turn into pd dataframe
    subject_info_df = pd.read_csv('~/Documents/MRes_Project/subject_info.csv')
    print(subject_info_df) # print current subject info

    # select a test patient from the information list
    test_subject_id = subject_info_df.Subject_ID[1]
    test_total_num_time_points = subject_info_df.Time_Points[1] # auto use all from excel sheet
    
    # select the time points we want to consider in analysis
    test_time_points_to_consider = [1,2,3,4,5]# for speed and testing the pipeline, only use the first two

    # define the type of registration we'd like to use
    registration_method = 'rigidfsl'

    # initialise a preprocess pipeline based on the test subject
    testProcess3DFLAIR = Process3DFLAIR(subject_info_df, test_subject_id, test_total_num_time_points, test_time_points_to_consider, registration_method, inter_subject = True) 

    
    # run variance pipeline
    out_file = "/home/ela/Documents/B-RAPIDD/B-RAP_0027/3D-FLAIR/variance_maps/rigid/intersubnormalised_all_timepoints_affine.nii.gz"
   
    testProcess3DFLAIR.runVariancePipeline(out_file)
    #testProcess3DFLAIR.calcVariance(out_file) # use this if all of the preprocessing has already occurred

     

    """
    # run subtraction
    out_file = "/home/ela/Documents/B-RAPIDD/B-RAP_0100/3D-FLAIR/rubbish.nii.gz"

    # minus non-ARIA from ARIA
    in_image_num = 3
    image_to_subtract_num = 1
    #testProcess3DFLAIR.subtractImages(in_image_num, image_to_subtract_num, out_file, threshold = True)
    testProcess3DFLAIR.runSubtraction(in_image_num, image_to_subtract_num, out_file)

    """

    # convert all of the temporal scans from DICOM to NIFTI
    #testProcess3DFLAIR.convertDICOMtoNIFTI() #NOTE: Edit this pipeline so that only files are converted if they don't exist

    # extract brains
   # testProcess3DFLAIR.extractBrainNIFTI(useT1 = False)
    #testProcess3DFLAIR.extractBrain()
  

    # correct bias field
    #testProcess3DFLAIR.correctBiasField(method = 'FSL')
    #testProcess3DFLAIR.intensityNormalisation(useBiasCorrected = True)