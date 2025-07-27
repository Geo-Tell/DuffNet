import os
import shutil

def copy_folder(source_folder, destination_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)

            # 使用shutil进行文件复制
            shutil.copy2(source_path, destination_path)
            # 如果需要保留源文件的元数据（如时间戳等），可以使用shutil.copy2

    print(f"复制完成，从 {source_folder} 到 {destination_folder}")

# 指定源文件夹和目标文件夹
source_folder = "/home/ch/Data/dataset_TfaSR/(60mor120m)to30m/DEM_Train/dem_9"
destination_folder = "/home/ch/HAT/datasets/DF2K_tif/DF2K_HR_sub"

# 调用复制函数
copy_folder(source_folder, destination_folder)
