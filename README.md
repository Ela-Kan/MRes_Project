# MRes_Project

## File structure for work:

### Structure of data
```
B-RAPIDD/ 									# folder with all images
├─ B-RAP_0027/ 									# specific patient file
│  ├─ 3D-FLAIR/ 
│  │  ├─ original_dicom/ 							# original dicom images
│  │  │  ├─ B-RAP_0027_01_D1/scans/12-RAPIDD_3D_FLAIR/resources/DICOM/files 	# time point 1 dicom (contains all DICOM files in the child folder)
│  │  │  ├─ B-RAP_0027_02_D1...							# time point 2 dicom (contains all DICOM files)
│  │  │  ├─ ...								# MRI sequence
│  │  ├─ original_nifti/ 							# original files converted to NIFTI
│  │  │  ├─ B-RAP_0027_01_D1.nii.gz 						# time point 1
│  │  │  ├─ B-RAP_0027_02_D1.nii.gz  						# time point 2
│  │  │  ├─ ...
│  │  │  ├─ json_info/								# folder containing all of the json information about the converted images
│  │  │	 │  ├─ B-RAP_0027_01_D1.json 						# time point 1
│  │  │	 │  ├─ B-RAP_0027_02_D1.json 						# time point 2
│  │  │	 │  ├─ ...
│  │  ├─ brain_nifti/ 								# brain extracted images
```

### Structure of code
```
~/Documents/MRes_Project/
├─ subject_info.csv				# spreadsheet of subject info (i.e. subject ID and number of time points)
├─ Code/ 					# folder containing all of the code
│  ├─ Preprocess3DFLAIR.py 			# class for preprocessing all of the images in the pipeline
```
