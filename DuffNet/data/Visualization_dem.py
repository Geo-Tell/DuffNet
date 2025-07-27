# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import rasterio
#
# # 打开.tif文件
# tif_path = '/home/ch/HAT/results/HAT_SRx4_ImageNet-pretrain/visualization/Set5_tif/dem0_0_HAT_SRx4_ImageNet-pretrain.TIF'
# elevation_data = rasterio.open(tif_path)
#
# # 读取高程数据
# elevation_array = elevation_data.read(1)
#
# # 获取地理转换信息
# transform = elevation_data.transform
#
# # 获取坐标轴范围
# x_range = transform[0] + transform[1] * elevation_data.width
# y_range = transform[3] + transform[5] * elevation_data.height
#
# # 创建X和Y的网格
# x = range(0, elevation_data.width)
# y = range(0, elevation_data.height)
# X, Y = plt.meshgrid(x, y)
#
# # 创建3D图形
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制三维地形图
# ax.plot_surface(X, Y, elevation_array, cmap='viridis')
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Elevation')
#
# # 显示图形
# plt.show()
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import rasterio
# import numpy as np
#
# # Open the .tif file
# tif_path = '/home/ch/HAT/results/HAT_SRx4_ImageNet-pretrain/visualization/Set5_tif/dem0_0_HAT_SRx4_ImageNet-pretrain.TIF'
# elevation_data = rasterio.open(tif_path)
#
# # Read elevation data
# elevation_array = elevation_data.read(1)
#
# # Get the geographical transformation information
# transform = elevation_data.transform
#
# # Get coordinate ranges
# x_range = transform[0] + transform[1] * elevation_data.width
# y_range = transform[3] + transform[5] * elevation_data.height
#
# # Create X and Y grids using NumPy meshgrid
# x = np.linspace(transform[0], x_range, elevation_data.width)
# y = np.linspace(transform[3], y_range, elevation_data.height)
# X, Y = np.meshgrid(x, y)
#
# # Create 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the 3D terrain
# ax.plot_surface(X, Y, elevation_array, cmap='viridis')
#
# # Set axis labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Elevation')
#
# # Show the plot
# plt.show()
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D

def visualize_elevation_map_3d(tif_path):
    # 打开 TIF 文件
    with rasterio.open(tif_path) as src:
        # 读取高程数据
        elevation_data = src.read(1)  # 使用1作为示例波段，多波段影像可能需要适当修改

    # 创建坐标网格
    x_size = elevation_data.shape[1]
    y_size = elevation_data.shape[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    X, Y = np.meshgrid(x, y)

    # 创建三维图形对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维高程图
    ax.plot_surface(X, Y, elevation_data, cmap='terrain')
    ax.set_zlabel('Elevation (meters)')
    ax.set_title('3D Elevation Map')

    plt.show()

# 指定高程图的TIF文件路径
#tif_file_path = "/home/ch/HAT/results/HAT_SRx4_ImageNet-pretrain/visualization/tfasr_dem/dem4_1122.TIF"
#tif_file_path = "/home/ch/HAT/results/HAT_SRx4_ImageNet-pretrain/visualization/Set5_tif/dem4_1122_HAT_SRx4_ImageNet-pretrain.TIF"
# tif_file_path = "/home/ch/HAT/datasets/Set5_tif/GTmod2/dem4_1122.TIF"
# 调用三维可视化函数
# visualize_elevation_map_3d(tif_file_path)
# samples = ['dem3_312.TIF', 'dem3_488.TIF', 'dem3_2693.TIF',
#            'dem3_717.TIF', 'dem3_389.TIF', 'dem3_2134.TIF']
from skimage import io
import tifffile
samples = ['dem197.TIF', 'dem1126.TIF', 'dem7323.TIF',
           'dem10631.TIF', 'dem707.TIF', 'dem15736.TIF',
           'dem15294.TIF']

for name in samples:
    gt_path = f'/home/ch/Data/pyrenees_2m/test/pyrenees_2m_test/{name}'
    img_gt = io.imread(gt_path)
    img_gt = img_gt * 1 + 0
    tifffile.imwrite(f'/home/ch/HAT/datasets/GT_py_new/gt_{name}', img_gt)

# samples = ['dem3_2134.TIF', 'dem3_312.TIF', 'dem3_2693.TIF',
#            'dem3_389.TIF', 'dem3_488.TIF', 'dem3_717.TIF']
# for name in samples:
#     gt_path = f'/home/ch/HAT/datasets/Set5_tif/GTmod2/{name}'
#     img_gt = io.imread(gt_path)
#     img_gt = img_gt * 1 + 0
#     tifffile.imwrite(f'/home/ch/HAT/datasets/tfa_visual_gt/gt_{name}', img_gt)

# import os
# import shutil
#
# # 源文件夹路径
# source_folder = '/home/ch/Data/pyrenees_2m/test/pyrenees_2m_test'
# # 新建的目标文件夹路径
# target_folder = '/home/ch/HAT/datasets/pyrenees_comparison_visual'
# # 指定要复制的文件名称
# specified_file_name = samples
#
# # 创建目标文件夹（如果不存在）
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
# for name in specified_file_name:
#     # 构造源文件的完整路径
#     source_file_path = os.path.join(source_folder, name)
#     # 构造目标文件的完整路径
#     target_file_path = os.path.join(target_folder, name)
#
#     # 检查源文件是否存在
#     if os.path.exists(source_file_path):
#         # 复制文件
#         shutil.copy(source_file_path, target_file_path)
#         print(f"文件 {specified_file_name} 已成功复制到 {target_folder} 文件夹中。")
#     else:
#         print(f"在 {source_folder} 文件夹中未找到名为 {specified_file_name} 的文件。")
