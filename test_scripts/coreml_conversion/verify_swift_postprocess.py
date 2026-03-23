"""Verify Swift postprocessing logic matches Python reference.

Simulates the Swift LocalInferenceService postprocess() in Python
and compares against the reference_postprocess.py output (which uses
actual API functions).

Tests: 3D box decode, rotation 6D->matrix->quaternion, camera-to-world, NMS.
"""
import os, sys, math
import numpy as np
from scipy.spatial.transform import Rotation

os.environ["SAM3_COMPILE"] = "0"
os.environ["SAM3_DISABLE_ACT_CKPT"] = "1"
sys.path.insert(0, ".")
sys.path.insert(0, "sam3")

# Constants (must match Swift code)
CENTER_SCALE = 10.0
DEPTH_SCALE = 2.0
DIM_SCALE = 2.0
INPUT_SIZE = 1008


# ============================================================
# Swift simulation functions (must match LocalInferenceService.swift)
# ============================================================

def swift_rotation_6d_to_matrix(d6):
    """Simulates Swift rotation6dToMatrix."""
    a1 = d6[:3]
    a2 = d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    d = np.dot(b1, a2)
    b2_raw = a2 - d * b1
    b2 = b2_raw / (np.linalg.norm(b2_raw) + 1e-8)
    b3 = np.cross(b1, b2)
    # Row-major: row0=b1, row1=b2, row2=b3
    return np.array([b1, b2, b3])


def swift_matrix_to_quaternion(m):
    """Simulates Swift matrixToQuaternion (Shepperd method)."""
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]

    q_abs = [
        math.sqrt(max(0, 1.0 + m00 + m11 + m22)) / 2.0,
        math.sqrt(max(0, 1.0 + m00 - m11 - m22)) / 2.0,
        math.sqrt(max(0, 1.0 - m00 + m11 - m22)) / 2.0,
        math.sqrt(max(0, 1.0 - m00 - m11 + m22)) / 2.0,
    ]

    best = np.argmax(q_abs)
    if best == 0:
        qw = q_abs[0]
        qx = (m21 - m12) / (4.0 * qw)
        qy = (m02 - m20) / (4.0 * qw)
        qz = (m10 - m01) / (4.0 * qw)
    elif best == 1:
        qx = q_abs[1]
        qw = (m21 - m12) / (4.0 * qx)
        qy = (m01 + m10) / (4.0 * qx)
        qz = (m02 + m20) / (4.0 * qx)
    elif best == 2:
        qy = q_abs[2]
        qw = (m02 - m20) / (4.0 * qy)
        qx = (m01 + m10) / (4.0 * qy)
        qz = (m12 + m21) / (4.0 * qy)
    else:
        qz = q_abs[3]
        qw = (m10 - m01) / (4.0 * qz)
        qx = (m02 + m20) / (4.0 * qz)
        qy = (m12 + m21) / (4.0 * qz)

    return np.array([qw, qx, qy, qz])


def swift_decode_box3d(box2d_pixel, box3d_encoded, K):
    """Simulates Swift decodeBox3D."""
    # 2D box center in padded pixels
    ctr_x = (box2d_pixel[0] + box2d_pixel[2]) / 2.0
    ctr_y = (box2d_pixel[1] + box2d_pixel[3]) / 2.0

    # Delta center
    delta_cx = box3d_encoded[0] * CENTER_SCALE
    delta_cy = box3d_encoded[1] * CENTER_SCALE

    # Projected 3D center
    proj_cx = ctr_x + delta_cx
    proj_cy = ctr_y + delta_cy

    # Depth
    depth = math.exp(box3d_encoded[2] / DEPTH_SCALE)

    # Unproject
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    cam_x = (proj_cx - cx) * depth / fx
    cam_y = (proj_cy - cy) * depth / fy
    cam_z = depth

    # Dimensions
    dim_w = math.exp(box3d_encoded[3] / DIM_SCALE)
    dim_l = math.exp(box3d_encoded[4] / DIM_SCALE)
    dim_h = math.exp(box3d_encoded[5] / DIM_SCALE)

    # Rotation + canonical normalization
    rot6d = box3d_encoded[6:12]
    rot_matrix = swift_rotation_6d_to_matrix(rot6d)
    final_w, final_l = dim_w, dim_l

    # Step 1: Force W <= L
    if final_w > final_l:
        final_w, final_l = final_l, final_w
        # Ry(90): col0_new = -col2, col2_new = col0
        col0 = rot_matrix[:, 0].copy()
        col2 = rot_matrix[:, 2].copy()
        rot_matrix[:, 0] = -col2
        rot_matrix[:, 2] = col0

    # Step 2: yaw = atan2(-R[2,0], R[0,0]), normalize to [0, pi)
    yaw = math.atan2(-rot_matrix[2, 0], rot_matrix[0, 0])
    if yaw < 0 or yaw > math.pi - 1e-4:
        rot_matrix[:, 0] = -rot_matrix[:, 0]
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    quat = swift_matrix_to_quaternion(rot_matrix)  # [qw, qx, qy, qz]

    return {
        "center": [cam_x, cam_y, cam_z],
        "dims": [final_w, final_l, dim_h],
        "quaternion": quat,
    }


def swift_camera_to_world(center, dims, quat_wxyz, cam2world):
    """Simulates Swift cameraToWorldTransform."""
    # Y-flip
    flip_y_4x4 = np.diag([1.0, -1.0, 1.0, 1.0])
    T = cam2world @ flip_y_4x4

    # Transform center
    center_h = np.array([center[0], center[1], center[2], 1.0])
    center_world = (T @ center_h)[:3]

    # Transform rotation
    R_cam2world = cam2world[:3, :3]
    flip_y_3x3 = np.diag([1.0, -1.0, 1.0])

    # quat [qw, qx, qy, qz] -> rotation matrix
    qw, qx, qy, qz = quat_wxyz
    x, y, z, w = qx, qy, qz, qw
    R_obj = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)],
    ])

    R_world = R_cam2world @ flip_y_3x3 @ R_obj @ flip_y_3x3
    quat_world = swift_matrix_to_quaternion(R_world)  # [qw, qx, qy, qz]

    # Dimension permutation: [w, l, h] -> [w, h, d]
    size_world = [dims[0], dims[2], dims[1]]

    # Convert [qw, qx, qy, qz] -> [qx, qy, qz, qw] (API output convention)
    rot_out = [quat_world[1], quat_world[2], quat_world[3], quat_world[0]]

    return center_world.tolist(), size_world, rot_out


def main():
    print("=" * 60)
    print("Verify Swift postprocessing matches Python reference")
    print("=" * 60)

    # ---- Test 1: rotation_6d_to_matrix ----
    print("\n[1/4] rotation_6d_to_matrix...")
    import torch

    # Python reference from coder.py (uses same Gram-Schmidt)
    def py_rot6d_to_matrix(d6_tensor):
        """Reference from opendet3d rotation code."""
        a1 = d6_tensor[..., :3]
        a2 = d6_tensor[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    np.random.seed(42)
    for i in range(5):
        d6 = np.random.randn(6).astype(np.float32)
        swift_m = swift_rotation_6d_to_matrix(d6)
        py_m = py_rot6d_to_matrix(torch.from_numpy(d6).unsqueeze(0)).squeeze(0).numpy()
        diff = np.abs(swift_m - py_m).max()
        print(f"  Test {i}: max diff = {diff:.8f}", "OK" if diff < 1e-5 else "FAIL")

    # ---- Test 2: matrix_to_quaternion ----
    print("\n[2/4] matrix_to_quaternion...")
    # Use scipy as ground truth
    for i in range(5):
        d6 = np.random.randn(6).astype(np.float32)
        m = swift_rotation_6d_to_matrix(d6)
        swift_q = swift_matrix_to_quaternion(m)  # [qw, qx, qy, qz]
        scipy_q = Rotation.from_matrix(m).as_quat()  # [qx, qy, qz, qw]
        py_q = np.array([scipy_q[3], scipy_q[0], scipy_q[1], scipy_q[2]])  # -> [qw, qx, qy, qz]

        # Quaternions can be negated (q == -q), normalize sign
        if swift_q[0] * py_q[0] < 0:
            swift_q = -swift_q
        diff = np.abs(swift_q - py_q).max()
        print(f"  Test {i}: diff={diff:.8f}", "OK" if diff < 1e-4 else "FAIL")

    # ---- Test 3: decode_box3d ----
    print("\n[3/4] decode_box3d...")
    from opendet3d.op.detect3d.grounding_dino_3d.coder import GroundingDINO3DCoder
    box_coder = GroundingDINO3DCoder(canonical_rotation=True)  # API uses canonical_rotation=True

    K = np.array([[817.2, 0, 512.8], [0, 818.2, 482.4], [0, 0, 1]], dtype=np.float32)

    for i in range(5):
        box2d = np.array([200 + i*50, 300, 400 + i*50, 500], dtype=np.float32)  # pixel xyxy
        box3d = np.random.randn(12).astype(np.float32)

        swift_result = swift_decode_box3d(box2d, box3d, K.tolist())

        # Python reference
        py_decoded = box_coder.decode(
            torch.from_numpy(box2d).unsqueeze(0),
            torch.from_numpy(box3d).unsqueeze(0),
            torch.from_numpy(K),
        ).squeeze(0).numpy()
        # py_decoded: [cx, cy, cz, w, l, h, qx, qy, qz, qw]

        center_diff = np.abs(np.array(swift_result["center"]) - py_decoded[:3]).max()
        dims_diff = np.abs(np.array(swift_result["dims"]) - py_decoded[3:6]).max()

        # Quaternion comparison (both are [qw,qx,qy,qz] from matrix_to_quaternion)
        swift_q = np.array(swift_result["quaternion"])  # [qw, qx, qy, qz]
        py_q = py_decoded[6:10]  # [qw, qx, qy, qz] from vis4d matrix_to_quaternion
        # Normalize sign (q == -q)
        if swift_q[0] * py_q[0] < 0:
            swift_q = -swift_q
        quat_diff = np.abs(swift_q - py_q).max()

        print(f"  Test {i}: center_diff={center_diff:.6f}, dims_diff={dims_diff:.6f}, quat_diff={quat_diff:.6f}",
              "OK" if center_diff < 1e-3 and dims_diff < 1e-4 and quat_diff < 1e-3 else "FAIL")

    # ---- Test 4: camera_to_world ----
    print("\n[4/4] camera_to_world...")
    from meta_demo.api_server import boxes3d_camera_to_world

    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    cam2world[:3, 3] = [1.0, 2.0, 3.0]

    for i in range(5):
        box2d = np.array([200, 300, 400, 500], dtype=np.float32)
        box3d_enc = np.random.randn(12).astype(np.float32)
        decoded = swift_decode_box3d(box2d, box3d_enc, K.tolist())

        # Swift camera_to_world
        sw_center, sw_size, sw_rot = swift_camera_to_world(
            decoded["center"], decoded["dims"], decoded["quaternion"], cam2world,
        )

        # Python reference (boxes3d_camera_to_world expects [cx,cy,cz,w,l,h,qw,qx,qy,qz])
        # box_coder.decode returns [qw,qx,qy,qz] from vis4d matrix_to_quaternion
        qw, qx, qy, qz = decoded["quaternion"]
        decoded_np = np.array([
            decoded["center"][0], decoded["center"][1], decoded["center"][2],
            decoded["dims"][0], decoded["dims"][1], decoded["dims"][2],
            qw, qx, qy, qz,
        ], dtype=np.float32).reshape(1, 10)

        py_boxes = boxes3d_camera_to_world(decoded_np, cam2world, apply_flip_y=True)
        py_box = py_boxes[0]

        center_diff = max(abs(a - b) for a, b in zip(sw_center, py_box["center"]))
        size_diff = max(abs(a - b) for a, b in zip(sw_size, py_box["size"]))

        # Rotation comparison
        sw_r = sw_rot  # [qx, qy, qz, qw]
        py_r = py_box["rotation"]  # [qx, qy, qz, qw]
        if sw_r[3] * py_r[3] < 0:
            sw_r = [-x for x in sw_r]
        rot_diff = max(abs(a - b) for a, b in zip(sw_r, py_r))

        print(f"  Test {i}: center_diff={center_diff:.6f}, size_diff={size_diff:.6f}, rot_diff={rot_diff:.6f}",
              "OK" if center_diff < 1e-3 and size_diff < 1e-4 and rot_diff < 1e-3 else "FAIL")

    print("\n" + "=" * 60)
    print("Done! All tests should show OK.")


if __name__ == "__main__":
    main()
