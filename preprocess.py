import logging
import os
import shutil
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *

class PreProcess:
    def __init__(self):
        # 配置logger
        logging.basicConfig(filename='/mnt/chenlb/datasets/ADNI/dcm2niiTools/log.log', 
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filemode='w')  # 'w'表示写入模式，每次运行程序会覆盖日志文件
        
        self.logger = logging.getLogger(__name__)
        
        
        self.label_file = pd.read_csv("/mnt/chenlb/datasets/ADNI/raw_data/adni2_axial_3d_3_class_6_22_2024.csv")
        self.id_to_group = self.label_file.set_index("Image Data ID")["Group"].to_dict()
        self.group_to_num = {"CN": 0, "MCI": 1, "AD": 2}
        self.target_shape = (45, 55, 45)# 目标形状
        self.problematic_samples = []
        self.shape_statistics = []
        
    def generate_nii_files(self, adni_path, output_path):
        self.raw_nii_path = os.path.join(output_path, "nii")
        if os.path.exists(self.raw_nii_path):
            shutil.rmtree(self.raw_nii_path)
        os.makedirs(self.raw_nii_path, exist_ok=True)
        subjects = os.listdir(adni_path)
        
        for subject in tqdm(subjects):
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
                        output_folder = os.path.join(self.raw_nii_path, os.path.basename(dcm_folder))
                        try:
                            process_dicom_to_nii(dcm_folder, output_folder, self.target_shape)
                        except Exception as e:
                            self.problematic_samples.append((dcm_folder, str(e)))
                            
        self.print_problematic_samples()
        
        
    def process(self, nii_folder, output_folder):
        self.skull_output_path = os.path.join(output_folder, "skull_stripping")
        if os.path.exists(self.skull_output_path):
            shutil.rmtree(self.skull_output_path)
        os.makedirs(self.skull_output_path, exist_ok=True)
        for nii_file in tqdm(os.listdir(nii_folder)):
            try:    
                magnitude_nii_file = os.path.join(nii_folder, nii_file, "magnitude.nii")
                if magnitude_nii_file.endswith(".nii") or nii_file.endswith(".nii.gz"):
                    output_file_path = os.path.join(self.skull_output_path, nii_file)
                    skull_stripping(magnitude_nii_file, output_file_path)
            except Exception as e:
                self.problematic_samples.append((nii_file, str(e)))
                    
        self.register_output_path = os.path.join(output_folder, "register")
        fixed_image_path = "/mnt/chenlb/datasets/utils/MNI152.nii" 
        
        if os.path.exists(self.register_output_path):
            shutil.rmtree(self.register_output_path)
        os.makedirs(self.register_output_path, exist_ok=True)
        
        for nii_file in tqdm(os.listdir(self.skull_output_path)):
            try:
                if "mask" not in nii_file:
                    input_path = os.path.join(self.skull_output_path, nii_file)
                    output_file_path = os.path.join(self.register_output_path, os.path.basename(nii_file).split('_')[0])
                    register_image(fixed_image_path, input_path, output_file_path)
            except Exception as e:
                self.problematic_samples.append((nii_file, str(e)))
                
                
        # self.bbox_path = os.path.join(output_folder, "bbox")
        # if not os.path.exists(self.bbox_path):
        #     shutil.rmtree(self.bbox_path)

        # os.makedirs(self.bbox_path, exist_ok=True)
        
        # for file_name in os.listdir(self.register_output_path):
        #     file_path = os.path.join(self.register_output_path, file_name)
        #     output_path = os.path.join(self.bbox_path, file_name)
            
        #     # 加载NIfTI文件
        #     img = nib.load(file_path)
        #     image_data = img.get_fdata()
            
        #     # 找到bounding box
        #     bbox = find_bounding_box(image_data)
        #     print(f'File: {file_name}, BBox: {bbox}')
            
        #     # 裁剪图像
        #     cropped_image = crop_image(image_data, bbox)
            
        #     # 保存裁剪后的图像
        #     cropped_img = nib.Nifti1Image(cropped_image, img.affine)
        #     nib.save(cropped_img, output_path)

        # bbox_list = []

        # for file_name in os.listdir(self.register_output_path):
        #     file_path = os.path.join(self.register_output_path, file_name)
            
        #     img = nib.load(file_path)
        #     image_data = img.get_fdata()
            
        #     bbox = find_bounding_box(image_data)
        #     bbox_list.append(bbox)

        # max_bbox = get_max_bounding_box(bbox_list)
        # self.logger.info(f'Maximum BBox: {max_bbox}')

        # for file_name in os.listdir(self.register_output_path):
        #     file_path = os.path.join(self.register_output_path, file_name)
        #     output_path = os.path.join(self.bbox_path, file_name)
            
        #     img = nib.load(file_path)
        #     image_data = img.get_fdata()
            
        #     cropped_image = crop_image(image_data, max_bbox)
            
        #     cropped_img = nib.Nifti1Image(cropped_image, img.affine)
        #     nib.save(cropped_img, output_path)

        self.print_problematic_samples()

    def generate_h5_files(self, nii_path, output_path):
        resized_save_path = os.path.join(output_path, "resized")
        if os.path.exists(resized_save_path):
            shutil.rmtree(resized_save_path)
        os.makedirs(resized_save_path, exist_ok=True)
        
        final_save_path = os.path.join(output_path, "generated_processed_45_55_45")
        if os.path.exists(final_save_path):
            shutil.rmtree(final_save_path)
        os.makedirs(final_save_path, exist_ok=True)
        
        data_entries = []
        nii_subjects = os.listdir(nii_path)
        total_samples = len(nii_subjects)
        
        pbar = tqdm(total=total_samples, desc="Processing NIfTI Samples")
        
        for subject in nii_subjects:
            subject_id = subject.split('.')[0]
            subject_path = os.path.join(nii_path, subject)
            label = self.get_label(subject_id)
            if label is None:
                continue
            try:
                original_3d_image, original_shape, resized_shape, original_affine = process_nii_to_3d_array(subject_path, self.target_shape, return_original_shape=True)
                self.shape_statistics.append({'subject': subject, 'original_shape': original_shape, 'resized_shape': resized_shape})
                dataset_name = os.path.basename(subject_path)
                data_entries.append({'dataset_name': dataset_name, 'data': original_3d_image, 'label': label})
                
                # Save resized NIfTI file with updated affine
                resized_nii_file_path = os.path.join(resized_save_path, subject)
                save_nifti_file(original_3d_image, original_affine, resized_nii_file_path)
            except Exception as e:
                self.problematic_samples.append((subject_path, str(e)))
            pbar.update(1)
        
        pbar.close()

        # Save shape statistics
        shape_statistics_df = pd.DataFrame(self.shape_statistics)
        shape_statistics_df.to_csv(os.path.join(final_save_path, "shape_statistics.csv"), index=False)

        # 按8:2的比例分为训练集和测试集
        train_entries, test_entries = train_test_split(data_entries, test_size=0.2, random_state=42)

        # 将训练集中标签为1的样本移动到测试集中
        train_entries_no_label1 = [entry for entry in train_entries if entry['label'] != 1]
        label1_entries = [entry for entry in train_entries if entry['label'] == 1]
        test_entries.extend(label1_entries)

        # Save data
        train_store_path = os.path.join(final_save_path, "train_data.h5")
        train_label_path = os.path.join(final_save_path, "train_labels.csv")
        test_store_path = os.path.join(final_save_path, "test_data.h5")
        test_label_path = os.path.join(final_save_path, "test_labels.csv")
        valid_store_path = os.path.join(final_save_path, "valid_data.h5")
        valid_label_path = os.path.join(final_save_path, "valid_labels.csv")
        self.save_data(train_entries_no_label1, train_store_path, train_label_path)
        self.save_data(test_entries, valid_store_path, valid_label_path)

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
            self.logger.error(f"Label not found for {subject_id}")
            return None

    def print_problematic_samples(self):
        self.logger.error("\nProblematic samples encountered during processing:")
        for sample, error in self.problematic_samples:
            self.logger.error(f"Sample: {sample} | Error: {error}")
            
            
if __name__ == '__main__':
    p = PreProcess()
    adni_path = "/mnt/chenlb/datasets/ADNI/raw_data/ADNI"  # input_path
    output_path = "/mnt/chenlb/datasets/ADNI"    # output_path
    # p.generate_nii_files(adni_path, output_path)
    # p.process("/mnt/chenlb/datasets/ADNI/nii", output_path)
    p.generate_h5_files("/mnt/chenlb/datasets/ADNI/register", output_path)