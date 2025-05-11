import numpy as np

depth_map = np.load('depth_map2.npy')  # Replace with your actual file name
print(depth_map)  # This will print the entire 480x640 array
depth_map[depth_map < 0] = 0  # Set all negative depths to 0
print("Min depth:", np.min(depth_map))
print("Max depth:", np.max(depth_map))
print("Average depth:", np.mean(depth_map))