import numpy as np

path = "/Users/jiacheng/Downloads/Case_AR-NUH014_cube_Index00001_102_179_106.npy"
image = np.load(path, mmap_mode="r", allow_pickle=False)

minimum = float(np.min(image))
maximum = float(np.max(image))
finite = bool(np.isfinite(image).all())
in_range = finite and minimum >= -1e-4 and maximum <= 1.0001

print("path:     ", path)
print("shape:    ", image.shape)
print("dtype:    ", image.dtype)
print("min:      ", minimum)
print("max:      ", maximum)
print("finite:   ", finite)
print("in [0,1]: ", in_range)