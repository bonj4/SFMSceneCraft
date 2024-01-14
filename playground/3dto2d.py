import numpy as np
from scipy.spatial.transform import Rotation


def world_to_2d_pixel(world_coordinates, rotation, translation, focal_length, distortion_k1, distortion_k2):
    for coord in world_coordinates:

        rotation_vector = np.array(rotation)
        rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()

        translation_vector = np.array(translation)

        intrinsic_matrix = np.array([[focal_length, 0, 0],
                                     [0, focal_length, 0],
                                     [0, 0, 1]])

        intrinsic_matrix[0, 0] *= (1 + distortion_k1)
        intrinsic_matrix[1, 1] *= (1 + distortion_k2)

        camera_coordinates = rotation_matrix.T @ (coord - translation_vector)

        pixel_coordinates_homogeneous = intrinsic_matrix @ camera_coordinates
        pixel_coordinates = (pixel_coordinates_homogeneous[:2] / pixel_coordinates_homogeneous[2]).astype(int)

        print(pixel_coordinates)


# Example usage for converting 3D to 2D
world_coordinates = [[-38.35719945 ,  1.05859846 , 19.96784492]]
rotation = [2.112527048686491, 1.1949149241028891, 0.9781286280531438]
translation = [-39.267631762708675, -0.017498039368377242, 18.930719402137353]
focal_length = 0.9631645519948918
distortion_k1 = 0.021481531813619496
distortion_k2 = -0.0011564140772048414

world_to_2d_pixel(world_coordinates, rotation, translation, focal_length, distortion_k1, distortion_k2)
