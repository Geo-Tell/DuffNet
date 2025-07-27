import os

# 输入文件夹路径
#folder_path = '/home/ch/HAT/datasets/DF2K/DF2K_bicx2_sub'  # 替换为你的图像文件夹路径
folder_path = '/home/ch/HAT/datasets/DF2K/DF2K_HR_sub'
# 获取文件夹中的所有文件
file_list = os.listdir(folder_path)

# 指定新的文件名长度
new_length = 5  # 替换为你希望的文件名长度

# 循环处理每个文件
for file_name in file_list:
    if file_name.endswith('.png'):  # 确保处理的是 PNG 文件
        file_path = os.path.join(folder_path, file_name)

        # 获取文件名和扩展名
        file_name_only, file_extension = os.path.splitext(file_name)

        # 如果文件名长度小于指定长度，用 '0' 填补在文件名前面
        while len(file_name_only) < new_length:
            file_name_only = '0' + file_name_only

        # 构建新的文件名
        new_file_name = file_name_only + file_extension

        # 如果新文件名与原文件名不同，进行重命名操作
        if new_file_name != file_name:
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"已修改文件名: {file_name} -> {new_file_name}")
    #break

print("文件名长度修改完成")
