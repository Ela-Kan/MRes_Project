"""
Main class to pre-process T1 MP-RAGE images within the pipeline for registration and subtraction of 3D FLAIR sequences

Author: Ela Kanani

Required dependencies: 
    * Chris Rorden's dcm2niiX version v1.0.20171215 (OpenJPEG build) GCC7.3.0 (64-bit Linux)

Optional dependencies:
    * 'pigz': allows for faster compression of images

"""
import numpy as np
import os
import pandas as pd
import shutil
import nipype.interfaces.fsl as fsl
# import the nibabel library so we can read in a nifti image
#import nibabel as nib
# import the BrainExtractor class
#from brainextractor import BrainExtractor


class PreprocessT1():

    def __init__(self, subject_id, num_time_points) -> None:
        """
        Initialises instance of the PreprocessT1 object.

        Parameters
        ----------
        subject_id : str
            string including current subject id. e.g. 'B-RAP_0027'
        
        num_time_points : int
            number of scans in the temporal series

            
        Returns
        -------
        None

        """
        
        # Define the global variables to be used throughout the implementations detailed.
        self.subject_id = subject_id # the subject id for the 3D FLAIR to be preprocessed
        self.num_time_points = num_time_points # the number of temporal scans for the given subject

        # create a folder tree with the appropriate file paths for use throughout this method, as according to accompanying doc
        self.subject_directory_root = '/home/ela/Documents/B-RAPIDD/' + self.subject_id +'/T1-MPRAGE/' # this folder should already exist with the downloaded data
        self.subject_dicom_directory = self.subject_directory_root + 'original_dicom/' 
     
        # create a folder for the converted NIFTI images and brain extractions
        self.subject_nifti_directory = self.subject_directory_root + 'original_nifti/'
        os.makedirs(self.subject_nifti_directory, exist_ok=True)
        self.subject_brain_directory = self.subject_directory_root + 'brain_nifti/'
        os.makedirs(self.subject_brain_directory, exist_ok=True)

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
        dicom_root_path = self.subject_dicom_directory + self.subject_id+ f'_{str(time_point).zfill(2)}_D1' # save the root path and fill the time_point with a 0

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
        TODO: EDIT THIS FUNCTION SO IT CAN COPE IF FILES ALREADY EXIST AND RENAME APPROPRIATELY

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
        #json_file_list = os.listdir(json_directory) # update nifti-only file list
        nifti_file_list.sort()
        nifti_file_list.pop()
        #json_file_list.sort()
        print(nifti_file_list)

        # count the number of files, each nifti has a corresponding json so this should be an equal number
        file_count = len(nifti_file_list)

        # rename the files in each folder to match the naming convention in the accompanying material
        #for index, (nifti_file_list, json_file_list) in enumerate(zip(nifti_file_list, json_file_list), start=1):
        for index, nifti_file_list in enumerate(nifti_file_list, start=1):
            new_file_name = self.subject_id+f'_{str(index).zfill(2)}_D1' # create string of subject ID and padded temporal (e.g. 01) reference, e.g. B-RAP_0027_01_D1
            new_nifti_name = f"{new_file_name}.nii.gz"
            #new_json_name = f"{new_file_name}.json"
            # rename the files
            os.rename(os.path.join(self.subject_nifti_directory, nifti_file_list), os.path.join(self.subject_nifti_directory, new_nifti_name))
            #os.rename(os.path.join(json_directory, json_file_list), os.path.join(json_directory, new_json_name))

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
        for i in range(self.num_time_points):
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

        return None
    



if __name__ == "__main__":

    # open the subject info table and turn into pd dataframe
    subject_info_df = pd.read_csv('~/Documents/MRes_Project/subject_info.csv')
    print(subject_info_df) # print current subject info

    # select a test patient from the information list
    test_subject_id = subject_info_df.Subject_ID[0]
    test_num_time_points = subject_info_df.Time_Points[0]
    
    # initialise a preprocess pipeline based on the test subject
    testPreprocessT1 = PreprocessT1(test_subject_id, test_num_time_points) 

    # convert all of the temporal scans from DICOM to NIFTI
    testPreprocessT1.convertDICOMtoNIFTI() #NOTE: Edit this pipeline so that only files are converted if they don't exist


    
