import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def pixel_to_3d_world(pixel_coordinates, rotation, translation, focal_length, distortion_k1, distortion_k2):
    for p in pixel_coordinates:
        rotation_vector = np.array(rotation)
        rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()

        translation_vector = np.array(translation)

        intrinsic_matrix = np.array([[focal_length, 0, 0],
                                     [0, focal_length, 0],
                                     [0, 0, 1]])

        intrinsic_matrix[0, 0] *= (1 + distortion_k1)
        intrinsic_matrix[1, 1] *= (1 + distortion_k2)

        pixel_homogeneous = np.array([p[0], p[1], 1])

        intermediate_result = np.linalg.inv(intrinsic_matrix) @ pixel_homogeneous

        world_homogeneous = translation_vector + (rotation_matrix @ intermediate_result)
        cam = np.array([0, 0, 0]).T
        cam_world = translation_vector + (rotation_matrix @ cam)

        vector = world_homogeneous - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        p3D = cam_world + 1.75 * unit_vector
        print("p3D : ", p3D)


pixel_coordinates = []


def mouse_callback(event, x, y, flags, param):
    global pixel_coordinates
    if param == 'image1' and event == cv2.EVENT_LBUTTONDOWN:
        pixel_coordinates.append([x, y])
        print(f"Captured Point on Image 1: ({x}, {y})")


path = r'/home/visio-ai/PycharmProjects/sfm/OpenSfM/data/berlin/images/03.jpg'
image = cv2.imread(path)
cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image 1', mouse_callback, param='image1')

cv2.imshow('Image 1', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Example usage
rotation = [2.112527048686491, 1.1949149241028891, 0.9781286280531438]
translation = [-39.267631762708675, -0.017498039368377242, 18.930719402137353]
focal_length = 0.9631645519948918
distortion_k1 = 0.021481531813619496
distortion_k2 = -0.0011564140772048414

pixel_to_3d_world(pixel_coordinates, rotation, translation, focal_length, distortion_k1, distortion_k2)
