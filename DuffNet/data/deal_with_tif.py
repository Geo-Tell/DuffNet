from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = '/home/ch/HAT/datasets/Set5_tif/GTmod2'  # 替换为你的原始 TIFF 图像文件夹路径
output_folder = '/home/ch/HAT/datasets/Set5_tif/LRbicx4'  # 替换为你希望保存下采样 TIFF 图像的文件夹路径

# 确保输出文件夹存在，如果不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

# 循环处理每个文件
for file_name in input_files:
    if 1:  # 确保处理的是 TIFF 文件
        file_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 打开 TIFF 图像
        img = Image.open(file_path)

        # 下采样
        new_width = img.width // 4
        new_height = img.height // 4
        downsampled_img = img.resize((new_width, new_height), Image.Resampling.NEAREST )
        # print(downsampled_img.show())
        # print(downsampled_img.size)

        # 保存下采样后的 TIFF 图像到输出文件夹
        downsampled_img.save(output_path, format='TIFF')
        #break

        print(f"已处理文件: {file_name}")

print("下采样完成")