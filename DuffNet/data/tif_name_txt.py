import os

# 输入文件夹路径
input_folder = '/home/ch/HAT/datasets/DF2K_tif/DF2K_HR_sub_debug'  # 替换为你的图像文件夹路径
output_txt_file = '/home/ch/HAT/hat/data/meta_info/meta_info_DF2Ksub_GT_tif_debug.txt'  # 替换为输出的txt文件路径

# 获取文件夹中的所有文件
input_files = os.listdir(input_folder)

# 打开txt文件以写入模式
with open(output_txt_file, 'w') as txt_file:
    # 循环处理每个文件
    for file_name in input_files:
        if 1:  # 确保处理的是图像文件
            # 添加字符"f"到文件名后面
            #new_file_name = f"{file_name.replace(' ', ' (256,256,3)')}"
            new_file_name = f"{file_name+' (64,64,1)'}"

            # 写入文件名到txt文件中
            txt_file.write(new_file_name + '\n')

            print(f"已写入文件名: {new_file_name}")
        #break

print("文件名写入txt文件完成")