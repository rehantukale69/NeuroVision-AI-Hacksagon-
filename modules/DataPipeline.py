import os
import random
from glob import glob

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

        # training transforms
        self.train_transforms = Compose([
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0)),
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

        # validation / test transforms
        self.test_transform = Compose([
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0)),
            ScaleIntensityd(keys="image")
        ])

        # inference transforms
        self.inference_transforms = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image")
        ])


    def LoadFilePaths(self, DatasetPath):

        patients = [
            p for p in os.listdir(DatasetPath)
            if os.path.isdir(os.path.join(DatasetPath, p))
        ]

        for patient in patients:

            patient_dir = os.path.join(DatasetPath, patient)

            item = {
                "image": [
                    os.path.join(patient_dir, "t1n.nii.gz"),
                    os.path.join(patient_dir, "t1c.nii.gz"),
                    os.path.join(patient_dir, "t2w.nii.gz"),
                    os.path.join(patient_dir, "t2f.nii.gz")
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

                for img in images:
                    if not os.path.exists(img):
                        raise Exception(f"Missing image: {img}")

                if not os.path.exists(label):
                    raise Exception(f"Missing label: {label}")

                img_volumes = [nib.load(img) for img in images]
                label_volume = nib.load(label)

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
            transform=self.train_transforms
        )

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
            data=self.CheckData(DataPathSet),
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

        nifti_files = glob(os.path.join(PatientDataPath, "*.nii")) + \
                      glob(os.path.join(PatientDataPath, "*.nii.gz"))

        dicom_files = glob(os.path.join(PatientDataPath, "*.dcm"))

        if len(nifti_files) > 0:

            return {"image": sorted(nifti_files)}

        elif len(dicom_files) > 0:

            return {"image": PatientDataPath}

        else:
            raise ValueError("No medical images found")