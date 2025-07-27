import rasterio
import numpy as np
import pyflwdir
# import geopandas as gpd

# from utils import (
#     quickplot,
#     colors,
#     cm,
#     plt,
# )  # data specific quick plot convenience method

# read elevation data of the rhine basin using rasterio
with rasterio.open("/home/ch/HAT/datasets/Set5_tif/GTmod2/dem0_1006.TIF", "r") as src:
    elevtn = src.read(1)
    nodata = src.nodata
    transform = src.transform
    crs = src.crs
    extent = np.array(src.bounds)[[0, 2, 1, 3]]
    latlon = src.crs.is_geographic
    prof = src.profile
# returns FlwDirRaster object
flw = pyflwdir.from_dem(
    data=elevtn,
    nodata=src.nodata,
    transform=transform,
    latlon=latlon,
    outlets="min",
)

feats = flw.streams(min_sto=10)
print(feats)
# gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
# create nice colormap of Blues with less white
# cmap_streams = colors.ListedColormap(cm.Blues(np.linspace(0.4, 1, 7)))
# gdf_plot_kwds = dict(column="strord", cmap=cmap_streams)
# # plot streams with hillshade from elevation data (see utils.py)
# ax = quickplot(
#     gdfs=[(gdf, gdf_plot_kwds)],
#     title="Streams based steepest gradient algorithm",
#     filename="flw_streams_steepest_gradient",
# )