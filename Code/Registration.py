"""
Main class to register multiple 3D FLAIR Images Depending on user input

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

class Registration():

    def __init__(self, reference_path, target_path, out_path) -> None:
        """
        Initialises instance of the Registration object.

        Parameters
        ----------
        reference_path : str
            string including the path of the reference (fixed) image 

        target_path : str
            string including the path of the target (moving) image 
            
        out_path : str
            string including the path of the output registered image 


            
        Returns
        -------
        None

        """

        # Define the global variables to be used throughout the implementations detailed.
        self.reference_path = Path(reference_path)
        self.target_path = Path(target_path)
        self.out_path = Path(out_path)

        return None

    def rigidFslFLIRT(self, cost_function, interpolation):
        """
        Performs rigid registration using the FSL FLIRT module.

        Parameters
        ----------
        cost_function : str
            string including the type of cost function desired (e.g. mutualinfo). See the FSL documentation for more options.

        interpolation : str
            string including the desired interpolation for the registration (e.g. trilinear). See the FSL documentation for more options.
            
            
        Returns
        -------
        flt_result : output of FLIRT registration.

        """     

        # need to register the two time points using rigid registration
        flt = fsl.FLIRT(
                        cost_func = 'mutualinfo', # inconsistent intensities 
                        dof = 6, # rigid body transformation
                        interp = 'trilinear')
        flt.inputs.in_file = self.target_path
        flt.inputs.reference = self.reference_path
        flt.inputs.output_type = "NIFTI_GZ" # save as a NIFTI file
        flt.inputs.out_file = self.out_path
        flt_result = flt.run() 

        return flt_result

    def affineFslFLIRT(self, cost_function, interpolation):
        """
        Performs affine registration using the FSL FLIRT module.

        Parameters
        ----------
        cost_function : str
            string including the type of cost function desired (e.g. mutualinfo). See the FSL documentation for more options.

        interpolation : str
            string including the desired interpolation for the registration (e.g. trilinear). See the FSL documentation for more options.
            
            
        Returns
        -------
        flt_result : output of FLIRT registration.

        """     

        # need to register the two time points using rigid registration
        flt = fsl.FLIRT(
                        cost_func = 'mutualinfo', # inconsistent intensities 
                        dof = 12, # affine transformation
                        interp = 'trilinear')
        flt.inputs.in_file = self.target_path
        flt.inputs.reference = self.reference_path
        flt.inputs.output_type = "NIFTI_GZ" # save as a NIFTI file
        flt.inputs.out_file = self.out_path
        flt_result = flt.run() 

        return flt_result

    def nonlinearFslFNIRT(self, affine_file_path):
        """
        Performs affine registration using the FSL FLIRT module.

        Parameters
        ----------
        cost_function : str
            string including the type of cost function desired (e.g. mutualinfo). See the FSL documentation for more options.

        interpolation : str
            string including the desired interpolation for the registration (e.g. trilinear). See the FSL documentation for more options.
            
            
        Returns
        -------
        flt_result : output of FLIRT registration.

        """     

        # need to register the two time points using rigid registration
        fnt = fsl.FNIRT(
                        affine_file = affine_file_path
                        )
        fnt.inputs.in_file = self.target_path
        fnt.inputs.ref_file = self.reference_path
        fnt.inputs.out_file = self.out_path
        fnt_result = fnt.run() 

        return fnt_result