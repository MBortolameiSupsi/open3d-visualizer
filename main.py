import open3d as o3d
import numpy as np
import cv2
# Define a callback function to add a cube
def add_cube_callback(vis):
    pos = (10,0,0)
    add_cube(vis, pos)

def add_cube(vis, pos):
    cube = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=100.0)
    cube.compute_vertex_normals()
    cube.translate(pos)  # Adjust position as needed
    vis.add_geometry(cube)
    vis.update_renderer()
    return False

def format_extrinsics(extrinsics):
    return "\n".join([
        " ".join(f"{val:+.2f}" if val >= 0 else f"{val:.2f}" for val in row)
        for row in extrinsics
    ])

def update_extrinsics_text(view_control):
    global previous_extrinsics

    # Get current extrinsics
    current_extrinsics = view_control.convert_to_pinhole_camera_parameters().extrinsic

    if not np.array_equal(current_extrinsics, previous_extrinsics):
        # Update previous extrinsics window
        prev_text = format_extrinsics(previous_extrinsics)
        prev_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        y0, dy = 100, 70
        for i, line in enumerate(prev_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(prev_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("Previous Extrinsics", prev_img)
        

        # Update current extrinsics window
        current_text = format_extrinsics(current_extrinsics)
        current_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for i, line in enumerate(current_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(current_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Current Extrinsics", current_img)
        

        # Update previous extrinsics
        previous_extrinsics = current_extrinsics.copy()

    cv2.waitKey(1)


# OpenCV window setup
cv2.namedWindow("Current Extrinsics", cv2.WINDOW_NORMAL)
cv2.namedWindow("Previous Extrinsics", cv2.WINDOW_NORMAL)
img_width=1000 
img_height=400
cv2.resizeWindow("Previous Extrinsics", img_width, img_height)
cv2.resizeWindow("Current Extrinsics", img_width, img_height)

previous_extrinsics = np.eye(4)


width= 2048
height= 1536
focal_length_mm = 16
pixel_size = 3.45 
downscale_factor = 2

width = int(width / downscale_factor)
height = int(height / downscale_factor)
pixel_size = pixel_size * downscale_factor
focal_length_pix = focal_length_mm / (pixel_size / 1000)
center = (width / 2, height / 2)

camera_matrix = np.array([[focal_length_pix, 0, center[0]],
    [0, focal_length_pix, center[1]], [0, 0, 1]])


camera_parameters = o3d.camera.PinholeCameraParameters()
camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera_matrix)
camera_parameters.extrinsic = np.eye(4)


# Set up the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=width, height=height)
# Register the callback for the "C" key
vis.register_key_callback(ord("C"), add_cube_callback)
cv2.moveWindow("Previous Extrinsics", width+100, 0)
cv2.moveWindow("Current Extrinsics", width+100, int(height/2))

# Apply the camera parameters to the view control
view_control = vis.get_view_control()
view_control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)


# Load the OBJ file
mesh = o3d.io.read_triangle_mesh(r"C:\Users\massimo.bortolamei\Documents\open3d-visualizer\tex_sample_03_aligned.obj")
# mesh = o3d.io.read_triangle_mesh(r"C:\Users\massimo.bortolamei\Documents\head-tracking\data\mapping\deca_scaled_translated\deca_scaled_translated.obj")
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500.0, origin=[0, 0, 0])
vis.add_geometry(axis)

counter = 0
while True:
    # if counter == 0:
    #     view_control.set_lookat([0, 0, 0])  # Adjust camera look-at point
    # elif counter == 1:
    #     view_control.set_up([0, 0, 1])  # Adjust the up direction of the camera
    # elif counter == 2:
    #     view_control.set_front([1, 0, 0])  # Adjust the front direction of the camera
    # elif counter == 3:
    #     view_control.set_zoom(1)  # Adjust the zoom level of the camera

    vis.poll_events()
    vis.update_renderer()
    update_extrinsics_text(view_control)
    # counter+=1
# vis.run()
# vis.destroy_window()

# Testing code
# my_extrinsics = np.array([
#     [0, 1, 0, 0],  # Rotation and translation for x
#     [0, 0, -1, 0],  # Rotation and translation for y
#     [-1, 0, 0, 1500],  # Rotation and translation for z
#     [0, 0, 0, 1]   # Homogeneous coordinate
# ])
# new_camera_parameters = camera_parameters
# new_camera_parameters.extrinsic = my_extrinsics
# view_control.convert_from_pinhole_camera_parameters(new_camera_parameters, allow_arbitrary=True)

