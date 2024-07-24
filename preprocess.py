import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import process_dicom_to_nii, process_nii_to_3d_array, save_nifti_file, load_nifti_file

class PreProcess:
    def __init__(self):
        self.label_file = pd.read_csv("/home/LAB/chenlb24/ADNI/adni2_axial_3d_3_class_6_22_2024.csv")
        self.id_to_group = self.label_file.set_index("Image Data ID")["Group"].to_dict()
        self.group_to_num = {"CN": 0, "MCI": 1, "AD": 2}
        self.train_store_path = "/home/LAB/chenlb24/ADNI/train_data.h5"
        self.valid_store_path = "/home/LAB/chenlb24/ADNI/valid_data.h5"
        self.test_store_path = "/home/LAB/chenlb24/ADNI/test_data.h5"
        self.train_label_path = "/home/LAB/chenlb24/ADNI/train_labels.csv"
        self.valid_label_path = "/home/LAB/chenlb24/ADNI/valid_labels.csv"
        self.test_label_path = "/home/LAB/chenlb24/ADNI/test_labels.csv"
        self.target_shape = (128, 128, 26)  # 目标形状
        self.problematic_samples = []
        self.shape_statistics_path = "/home/LAB/chenlb24/ADNI/shape_statistics.csv"
        self.resized_nii_path = "/home/LAB/chenlb24/ADNI/resized_nii"  # Resized NIfTI files directory
        os.makedirs(self.resized_nii_path, exist_ok=True)
        self.shape_statistics = []

    def generate_nii_files(self, adni_path, nii_path):
        subjects = os.listdir(adni_path)
        
        for subject in subjects:
            subject_path = os.path.join(adni_path, subject)
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
                        output_folder = os.path.join(nii_path, os.path.basename(dcm_folder))
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
                original_3d_image, original_shape, resized_shape, original_affine = process_nii_to_3d_array(subject_path, self.target_shape, return_original_shape=True)
                self.shape_statistics.append({'subject': subject, 'original_shape': original_shape, 'resized_shape': resized_shape})
                dataset_name = os.path.basename(subject_path)
                data_entries.append({'dataset_name': dataset_name, 'data': original_3d_image, 'label': label})
                
                # Save resized NIfTI file with updated affine
                resized_nii_file_path = os.path.join(self.resized_nii_path, f"{dataset_name}_resized.nii")
                save_nifti_file(original_3d_image, original_affine, resized_nii_file_path)
            except Exception as e:
                self.problematic_samples.append((subject_path, str(e)))
            pbar.update(1)
        
        pbar.close()

        # Save shape statistics
        shape_statistics_df = pd.DataFrame(self.shape_statistics)
        shape_statistics_df.to_csv(self.shape_statistics_path, index=False)

        # 按8:2的比例分为训练集和测试集
        train_entries, test_entries = train_test_split(data_entries, test_size=0.2, random_state=42)

        # 将训练集中标签为1的样本移动到测试集中
        train_entries_no_label1 = [entry for entry in train_entries if entry['label'] != 1]
        label1_entries = [entry for entry in train_entries if entry['label'] == 1]
        test_entries.extend(label1_entries)

        # Save data
        self.save_data(train_entries_no_label1, self.train_store_path, self.train_label_path)
        self.save_data(test_entries, self.test_store_path, self.valid_label_path)

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
    adni_path = "/home/LAB/chenlb24/ADNI/ADNI"  # input_path
    nii_path = "/home/LAB/chenlb24/ADNI/nii"    # output_path
    # p.generate_nii_files(adni_path, nii_path)
    p.generate_h5_files(nii_path)