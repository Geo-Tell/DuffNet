from PIL import Image
import os

# 输入和输出文件夹路径
# input_folder = '/home/ch/HAT/datasets/DF2K/DF2K_HR_sub'  # 替换为你的原始图像文件夹路径
# output_folder = '/home/ch/HAT/datasets/DF2K/DF2K_HR_sub_splitting'  # 替换为你希望保存分割图像的文件夹路径

input_folder = '/home/ch/HAT/datasets/Set5/GTmod2'  # 替换为你的原始图像文件夹路径
output_folder = '/home/ch/HAT/datasets/Set5/GTmod2_splitting'  # 替换为你希望保存分割图像的文件夹路径

# 确保输出文件夹存在，如果不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

# 循环处理每个文件
for file_name in input_files:
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 确保处理的是图像文件
        file_path = os.path.join(input_folder, file_name)

        # 打开图像
        img = Image.open(file_path)

        # 获取图像大小
        width, height = img.size

        # 计算分割后的子图像大小
        new_width = width // 2
        new_height = height // 2

        # 分割图像为四张子图像
        sub_images = [
            img.crop((0, 0, new_width, new_height)),
            img.crop((new_width, 0, width, new_height)),
            img.crop((0, new_height, new_width, height)),
            img.crop((new_width, new_height, width, height))
        ]

        # 为分割后的子图像命名并保存
        for i, sub_img in enumerate(sub_images, start=1):
            output_file_name = f"{file_name.replace('.', f'_s{i}.')}"
            output_path = os.path.join(output_folder, output_file_name)
            sub_img.save(output_path)

            print(f"已保存分割图像: {output_file_name}")
    #break

print("图像分割完成")
