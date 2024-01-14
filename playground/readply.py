import numpy as np
from plyfile import PlyData, PlyElement


def read_ply(file_path):
    with open(file_path, 'rb') as f:
        ply_data = PlyData.read(f)
    return ply_data['vertex']


def get_points_around_given_point(points, given_point, radius):
    # Extract x, y, z coordinates of given point
    x, y, z = given_point
    point_x = points['x']
    point_y = points['y']
    point_z = points['z']
    # Calculate the Euclidean distance from each point to the given point
    distances = np.sqrt((point_x - x) ** 2 + (point_y - y) ** 2 + (point_z - z) ** 2)

    # Find indices of points within the specified radius
    indices = np.where(distances <= radius)[0]

    # Extract points within the radius
    points_around_given_point =[[x,y,z] for x,y,z in zip(point_x[indices],point_y[indices],point_z[indices])]

    return points_around_given_point

given_point = [14.041853411893493,
               22.26605894683144,
               15.57962347264628]

raw3d = [-38.35157078, 1.04474809, 19.97712537]
reference_lla = {
    "latitude": 52.51896296296295,
    "longitude": 13.40037685185185,
    "altitude": 0.0
}
gps_position = [
    4.776831919437688,
    9.376193488755234,
    38.99999132193625
]
gps = {
    "latitude": 52.51904722222222,
    "longitude": 13.400447222222223,
    "altitude": 39.0,
    "dop": 5.0
}

# Replace with the desired radius
radius = 1.0
ply_file_path = '/home/visio-ai/PycharmProjects/sfm/OpenSfM/data/berlin/undistorted/depthmaps/merged.ply'

# Read PLY file
ply_data = read_ply(ply_file_path)

# Get points around the given point
points_around_given_point = get_points_around_given_point(ply_data, given_point, radius)

# Print the result
print("Points around the given point:")
print(points_around_given_point)
if given_point in points_around_given_point: print("uyes")