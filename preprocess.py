import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import process_dicom_to_nii, process_nii_to_3d_array

class PreProcess:
    def __init__(self):
        self.label_file = pd.read_csv("/home/LAB/chenlb24/adni/adni2_axial_3d_3_class_6_22_2024.csv")
        self.id_to_group = self.label_file.set_index("Image Data ID")["Group"].to_dict()
        self.group_to_num = {"CN": 0, "MCI": 1, "AD": 2}
        self.train_store_path = "/home/LAB/chenlb24/adni/train_data.h5"
        self.valid_store_path = "/home/LAB/chenlb24/adni/valid_data.h5"
        self.test_store_path = "/home/LAB/chenlb24/adni/test_data.h5"
        self.train_label_path = "/home/LAB/chenlb24/adni/train_labels.csv"
        self.valid_label_path = "/home/LAB/chenlb24/adni/valid_labels.csv"
        self.test_label_path = "/home/LAB/chenlb24/adni/test_labels.csv"
        self.target_shape = (96, 96, 64)  # 目标形状，其中64为高度
        self.problematic_samples = []

    def generate_nii_files(self, path):
        subjects = os.listdir(path)
        
        for subject in subjects:
            subject_path = os.path.join(path, subject)
            methods = os.listdir(subject_path)
            for method in methods:
                if method != "Field_Mapping":
                    continue
                method_path = os.path.join(subject_path, method)
                visit_times = os.listdir(method_path)
                for visit_time in visit_times:
                    time_path = os.path.join(method_path, visit_time)
                    image_datas = os.listdir(time_path)
                    for image_data in image_datas:
                        dcm_folder = os.path.join(time_path, image_data)
                        output_folder = f"/home/LAB/chenlb24/adni/nii/{os.path.basename(dcm_folder)}"
                        try:
                            process_dicom_to_nii(dcm_folder, output_folder, self.target_shape)
                        except Exception as e:
                            self.problematic_samples.append((dcm_folder, str(e)))

        self.print_problematic_samples()

    def generate_h5_files(self, nii_path):
        data_entries = []
        nii_subjects = os.listdir(nii_path)
        total_samples = len(nii_subjects)
        
        pbar = tqdm(total=total_samples, desc="Processing NIfTI Samples")
        
        for subject in nii_subjects:
            subject_path = os.path.join(nii_path, subject)
            label = self.get_label(subject)
            if label is None:
                continue
            try:
                preprocessed_3d_image = process_nii_to_3d_array(subject_path, self.target_shape)
                dataset_name = os.path.basename(subject_path)
                data_entries.append({'dataset_name': dataset_name, 'data': preprocessed_3d_image, 'label': label})
            except Exception as e:
                self.problematic_samples.append((subject_path, str(e)))
            pbar.update(1)
        
        pbar.close()

        train_entries, test_entries = train_test_split(data_entries, test_size=0.2, random_state=42)
        train_entries, valid_entries = train_test_split(train_entries, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

        self.save_data(train_entries, self.train_store_path, self.train_label_path)
        self.save_data(valid_entries, self.valid_store_path, self.valid_label_path)
        self.save_data(test_entries, self.test_store_path, self.test_label_path)

        self.print_problematic_samples()

    def save_data(self, entries, data_path, label_path):
        with h5py.File(data_path, 'w') as data_store:
            labels = []
            for entry in entries:
                dataset_name = entry['dataset_name']
                data = entry['data']
                label = entry['label']
                data_store.create_dataset(dataset_name, data=data)
                labels.append({'dataset_name': dataset_name, 'label': label})
            labels_df = pd.DataFrame(labels)
            labels_df.to_csv(label_path, index=False)
        
    def get_label(self, subject_id):
        try:
            group = self.id_to_group[subject_id]
            label = self.group_to_num[group]
            return label
        except IndexError:
            print(f"Label not found for {subject_id}")
            return None

    def print_problematic_samples(self):
        print("\nProblematic samples encountered during processing:")
        for sample, error in self.problematic_samples:
            print(f"Sample: {sample} | Error: {error}")

if __name__ == '__main__':
    p = PreProcess()
    # p.generate_nii_files("/home/LAB/chenlb24/adni/ADNI")
    p.generate_h5_files("/home/LAB/chenlb24/adni/nii")
