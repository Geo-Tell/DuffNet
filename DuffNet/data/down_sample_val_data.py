from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = '/home/ch/HAT/datasets/Set5/GTmod2_splitting'  # 替换为你的原始图像文件夹路径
output_folder = '/home/ch/HAT/datasets/Set5/LRbicx2_splitting'  # 替换为你希望保存下采样图像的文件夹路径

# 确保输出文件夹存在，如果不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

# 循环处理每个文件
for file_name in input_files:
    if file_name.endswith('.png'):  # 确保处理的是 PNG 文件
        file_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 打开图像
        img = Image.open(file_path)

        # 下采样两倍
        width, height = img.size
        new_width = width // 4
        new_height = height // 4
        downsampled_img = img.resize((new_width, new_height), Image.LANCZOS )

        # 保存下采样后的图像到输出文件夹
        downsampled_img.save(output_path)

        print(f"已处理文件: {file_name}")

print("下采样完成")
