import numpy as np

# Robot Height
robot_height = 10.9

# Outpost 1 Camera position
o1_point = (-8.98976, 177.5661, 457.6042)

# Outpost 1 Camera to World Matrix
c2w_matrix_1 = np.array([[-0.70486, 0.22467, -0.67282, -8.98975],
                         [-0.02764, 0.93910, 0.34254, 177.56610],
                         [-0.70880, -0.26004, 0.65573, 457.60420],
                         [0.00000, 0.00000, 0.00000, 1.00000]])

# Outpost 1 Camera Projection Matrix
projection_inv_1 = np.array([[0.87706,	0.00000, 0.00000, 0.00000],
                                [0.00000,	0.49335, 0.00000, 0.00000],
                                [0.00000,	0.00000, 0.00000, -1.00000],
                                [0.00000,	0.00000, -1.66617, 1.66717]])

# Outpost 3 Camera position
o3_point = (817.1082, 177.5661, -9.41583)

# Outpost 3 Camera to World Matrix
c2w_matrix_3 = np.array([[0.69863, -0.23571, 0.67554, 817.10820],
                         [-0.03390, 0.93221, 0.36032, 177.56610],
                         [0.71468, 0.27463, -0.64328, -9.41580],
                         [0.00000, 0.00000, 0.00000, 1.00000]])

# Outpost 3 Camera Projection Matrix Inverse
projection_inv_3 = np.array([[0.87706,	0.00000, 0.00000, 0.00000],
                                [0.00000,	0.49335, 0.00000, 0.00000],
                                [0.00000,	0.00000, 0.00000, -1.00000],
                                [0.00000,	0.00000, -1.66617, 1.66717]])


s2w_matrix1 = np.dot(c2w_matrix_1, projection_inv_1)
s2w_matrix3 = np.dot(c2w_matrix_3, projection_inv_3)


def s2r(screen_point, camera_type):
    """
    This function is used to convert screen point into world point
    The screen to world matrix s2w_matrix needs to specify
    Also, the camera's world coordinate needs to specify
    """

    # based on camera type, determine the camera matrix
    if camera_type == "o1":
        s2w_matrix = s2w_matrix1
        camera_point = o1_point
    elif camera_type == "o3":
        s2w_matrix = s2w_matrix3
        camera_point = o3_point
    else:
        return

    # convert screen point to world point on far clipping plane
    (sx, sy) = screen_point
    plane_point = np.array([(sx*2-1)*1000, ((1-sy)*2-1)*1000, 1000, 1000])
    world_point = np.dot(s2w_matrix, plane_point)
    world_point = world_point[:-1].tolist()

    # convert point on far clipping plane to the location of robots
    (x1, y1, z1) = camera_point
    (x2, y2, z2) = world_point
    a1 = y1 - robot_height
    a2 = y1 - y2
    xr = x1 + a1/a2 * (x2-x1)
    zr = z1 + a1/a2 * (z2-z1)

    return xr, zr
