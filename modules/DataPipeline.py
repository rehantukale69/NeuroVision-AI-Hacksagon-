# Load Dataset all files path -> check all stuff exists ->Load Data -> Tarnsform for training Done

import os
from glob import glob
import random
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandRotate90d
)
import nibabel as nib
 


class PipeLine:
    def __init__(self):
        self.files = []

        self.train_transforms = Compose([
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(keys=["image","label"], pixdim=(1,1,1)),
            ScaleIntensityd(keys="image"),
            RandCropByPosNegLabeld(
                keys=["image","label"],
                spatial_size=(96,96,96),
                pos=1,
                neg=1,
                num_samples=4
            ),
        RandFlipd(keys=["image","label"], prob=0.5),
        RandRotate90d(keys=["image","label"], prob=0.5)
        ])

        self.test_transform = Compose([
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(keys=["image","label"], pixdim=(1,1,1)),
            ScaleIntensityd(keys="image")
        ])
        
        self.inference_transforms = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image")
        ])

    def LoadFilePaths(self,  DatasetPath):
        patients = sorted(os.listdir(DatasetPath))

        
       
        for patient in patients:
            patient_dir = os.path.join(DatasetPath, patient)
            
            item = {
                "image": [
                    os.path.join(patient_dir, "t1n.nii.gz"),
                    os.path.join(patient_dir, "t1c.nii.gz"),
                    os.path.join(patient_dir, "t2w.nii.gz"),
                    os.path.join(patient_dir, "t2f.nii.gz"),
                    ],
                    
                "label": os.path.join(patient_dir, "seg.nii.gz")
                }
            self.files.append(item)

        random.shuffle(self.files)
        
        self.train_size = int(0.7 * len(self.files))
        self.val_size = int(0.15 * len(self.files))
        
    
    def GetTrainPaths(self):
        return self.files[:self.train_size]
    
    def GetTestPaths(self):
        return self.files[self.train_size:self.train_size+self.val_size]

    def GetValidPaths(self):
        return self.files[self.train_size+self.val_size:]
    
    def CheckData(self, DataPathSet):
        
        valid_data = []
        
        for item in DataPathSet:
            
            images = item["image"]
            label = item["label"]
            
            try:
                # check files exist
                for img in images:
                    if not os.path.exists(img):
                        raise Exception(f"Missing image: {img}")
                    
                if not os.path.exists(label):
                    raise Exception(f"Missing label: {label}")

                # load images
                img_volumes = [nib.load(img) for img in images]
                label_volume = nib.load(label)

                # check shapes
                img_shapes = [img.shape for img in img_volumes]
                
                if not all(s == img_shapes[0] for s in img_shapes):
                    raise Exception("Modalities have different shapes")
                
                if label_volume.shape != img_shapes[0]:
                    raise Exception("Label shape mismatch")
                
                valid_data.append(item)

            except Exception as e:
                print("Invalid sample skipped:", e)
                
        return valid_data
    

    def LoadTrainData(self, DataPathSet, batch_size=2, shuffle=True):
        
        dataset = Dataset(
             data=self.CheckData(DataPathSet),
             transform=self.train_transforms)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        return loader


    def LoadTestData(self, DataPathSet):
        
        dataset = Dataset(
            data=DataPathSet,
            transform=self.test_transform
            )
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return loader
    
    def LoadPatientData(self, PatientDataPath):
        # find nifti files
        nifti_files = glob(os.path.join(PatientDataPath, "*.nii")) 
        glob(os.path.join(PatientDataPath, "*.nii.gz"))
        
        # find dicom files
        dicom_files = glob(os.path.join(PatientDataPath, "*.dcm"))
        
        # Case 1: NIfTI dataset
        if len(nifti_files) > 0:
            patient_data = {
                "image": sorted(nifti_files)
            }
            return patient_data
        
        # Case 2: DICOM dataset
        elif len(dicom_files) > 0:
            patient_data = {
                "image": PatientDataPath   # MONAI loads full DICOM series
                }
            return patient_data
        else:
            raise ValueError(f"No medical images found in {PatientDataPath}")
    