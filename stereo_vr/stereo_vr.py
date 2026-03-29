import argparse
import numpy as np
import cv2
import os


def equirectangular_to_perspective(img, fov_deg, yaw_deg, pitch_deg, out_w, out_h, projection='rectilinear', vfov_deg=None, input_hfov_deg=360.0, input_vfov_deg=180.0):
    h_e, w_e = img.shape[:2]
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Use angle-based mapping per pixel so we can support wide FOVs.
    hfov = fov
    if vfov_deg is not None:
        vfov = np.deg2rad(vfov_deg)
    else:
        # derive vertical fov from aspect ratio for rectilinear default
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * (out_h / out_w))


    i, j = np.meshgrid(np.arange(out_w), np.arange(out_h))
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    # map pixels linearly to angles in camera space
    if projection == 'spherical':
        # In spherical (angular) mode, map pixel indices directly to angular ranges.
        # alpha: -hfov/2 .. +hfov/2 across i=0..W-1
        # beta:  +vfov/2 .. -vfov/2 across j=0..H-1 (top to bottom)
        alpha = (i.astype(np.float64) / (out_w - 1)) * hfov - hfov / 2.0
        beta = (1.0 - j.astype(np.float64) / (out_h - 1)) * vfov - vfov / 2.0
    else:
        # rectilinear-like angular mapping (suitable for perspective-like rendering)
        alpha = (i - cx) / cx * (hfov / 2.0)
        beta = -(j - cy) / cy * (vfov / 2.0)

    # direction vector for each pixel using spherical angles (alpha: azimuth, beta: elevation)
    x = np.sin(alpha) * np.cos(beta)
    y = np.sin(beta)
    z = np.cos(alpha) * np.cos(beta)

    vec = np.stack((x, y, z), axis=-1)

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = Ry @ Rx

    # rotated vectors for remapping (used to compute lon/lat -> pixel coords)
    vec_rot_for_map = vec @ R.T

    # Correct orientation for point cloud only: rotate vectors by +90 degrees about X
    Rcorr = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    vec_rot_for_points = vec_rot_for_map @ Rcorr.T

    lon = np.arctan2(vec_rot_for_map[..., 0], vec_rot_for_map[..., 2])
    lat = np.arcsin(np.clip(vec_rot_for_map[..., 1], -1.0, 1.0))

    # Map lon/lat to input image coordinates using the input image angular coverage.
    input_hfov = np.deg2rad(input_hfov_deg)
    input_vfov = np.deg2rad(input_vfov_deg)

    lon_min = -input_hfov / 2.0
    lon_max = input_hfov / 2.0
    lat_max = input_vfov / 2.0
    lat_min = -input_vfov / 2.0

    u = (lon - lon_min) / (lon_max - lon_min) * (w_e - 1)
    v = (lat_max - lat) / (lat_max - lat_min) * (h_e - 1)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    persp = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # Also return the rotated direction vectors (camera-space rays) for triangulation
    # (use the orientation-corrected vectors for point cloud generation)
    return persp, vec_rot_for_points


def compute_disparity(left, right, num_disp=128, block_size=9):
    min_disp = 0
    num_disp = (num_disp // 16) * 16
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block_size,
                                   P1=8 * 3 * block_size ** 2,
                                   P2=32 * 3 * block_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    return disp


def disparity_to_pointcloud(disp, left_color, left_vec, right_vec, baseline, f_px=None, mask=None):
    # Compute 3D points by scaling unit direction vectors by depth derived from disparity.
    # left_vec/right_vec: (H,W,3) direction vectors in camera space.
    h, w = disp.shape
    if mask is None:
        mask = disp > 0
    valid = mask & (disp > 0)

    if f_px is not None:
        disp_valid = disp[valid]
        # avoid division by zero
        disp_valid[disp_valid <= 0] = np.nan
        # depth along camera forward (approximate): Z = f * B / disp
        Z = (f_px * float(baseline)) / disp_valid

        dirs = left_vec[valid].astype(np.float64)
        norms = np.linalg.norm(dirs, axis=1)
        norms[norms == 0] = 1.0
        unit_dirs = dirs / norms[:, None]

        pts = unit_dirs * Z[:, None]
        # flip Z so +Z points forward in viewer conventions
        pts[:, 2] *= -1.0
        colors = left_color[valid]
        # remove any NaN or inf points
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        colors = colors[finite_mask]
        return pts, colors

    # Fallback: triangulation (previous method) if f_px is not provided
    u_dirs = left_vec[valid].astype(np.float64)
    v_dirs = right_vec[valid].astype(np.float64)

    O_l = np.zeros(3, dtype=np.float64)
    O_r = np.array([baseline, 0.0, 0.0], dtype=np.float64)

    uu = np.einsum('ij,ij->i', u_dirs, u_dirs)
    vv = np.einsum('ij,ij->i', v_dirs, v_dirs)
    uv = np.einsum('ij,ij->i', u_dirs, v_dirs)

    OrOl = O_r - O_l
    b1 = np.dot(u_dirs, OrOl)
    b2 = np.dot(v_dirs, OrOl)

    denom = uu * vv - uv * uv

    eps = 1e-6
    valid_den = np.abs(denom) > eps

    t = np.zeros_like(denom)
    s = np.zeros_like(denom)
    t[valid_den] = (b1[valid_den] * vv[valid_den] + b2[valid_den] * uv[valid_den]) / denom[valid_den]
    s[valid_den] = (b2[valid_den] * uu[valid_den] + b1[valid_den] * uv[valid_den]) / denom[valid_den]

    if np.any(~valid_den):
        idx = np.nonzero(~valid_den)[0]
        u_f = u_dirs[idx]
        v_f = v_dirs[idx]
        alpha_u = np.arctan2(u_f[:, 0], u_f[:, 2])
        alpha_v = np.arctan2(v_f[:, 0], v_f[:, 2])
        tan_diff = np.tan(alpha_u) - np.tan(alpha_v)
        tan_diff[np.abs(tan_diff) < 1e-8] = 1e-8
        Z = baseline / tan_diff
        uz = u_f[:, 2]
        uz[np.abs(uz) < 1e-8] = 1e-8
        t_f = Z / uz
        t[idx] = t_f
        s[idx] = 0.0

    P_l = u_dirs * t[:, None] + O_l
    P_r = v_dirs * s[:, None] + O_r

    pts = (P_l + P_r) / 2.0
    # flip Y to match viewer forward convention
    pts[:, 1] *= -1.0
    colors = left_color[valid]
    return pts, colors


def save_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(verts, colors):
            f.write(f'{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n')


def save_obj_quads(filename, verts_grid, colors_grid, valid_mask=None, quad_size=0.01):
    """Write an OBJ where each valid point is rendered as a small quad.

    - verts_grid: (H,W,3) float array of vertex positions
    - colors_grid: (H,W,3) uint8 BGR colors (0-255)
    - valid_mask: optional (H,W) boolean mask of valid vertices; if None, any finite vertex is valid
    - quad_size: world-space side length of each quad (default 0.01)
    """
    H, W, _ = verts_grid.shape
    verts_flat = verts_grid.reshape(-1, 3)
    cols_flat = colors_grid.reshape(-1, 3)

    if valid_mask is None:
        good = np.isfinite(verts_flat).all(axis=1)
    else:
        good = valid_mask.reshape(-1)

    with open(filename, 'w') as f:
        f.write('# OBJ with per-point quads and vertex colors (v x y z r g b)\n')

        vertex_counter = 1

        for v, col, ok in zip(verts_flat, cols_flat, good):
            if not ok:
                continue
            if not np.isfinite(v).all():
                continue

            # compute an approximate normal (from origin to vertex)
            n = v.astype(np.float64)
            nrm = np.linalg.norm(n)
            if nrm == 0 or np.isnan(nrm):
                continue
            n = n / nrm

            # pick a stable 'up' vector to build a tangent basis
            up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
            if abs(np.dot(up, n)) > 0.99:
                up = np.array((1.0, 0.0, 0.0), dtype=np.float64)

            t1 = np.cross(up, n)
            t1n = np.linalg.norm(t1)
            if t1n == 0 or np.isnan(t1n):
                continue
            t1 = t1 / t1n
            t2 = np.cross(n, t1)

            half = float(quad_size) * 0.5

            p1 = v - t1 * half - t2 * half
            p2 = v + t1 * half - t2 * half
            p3 = v + t1 * half + t2 * half
            p4 = v - t1 * half + t2 * half

            # write four vertices with color (convert BGR->RGB in 0..1 range)
            r = float(col[2]) / 255.0
            g = float(col[1]) / 255.0
            b = float(col[0]) / 255.0

            f.write(f'v {p1[0]} {p1[1]} {p1[2]} {r} {g} {b}\n')
            f.write(f'v {p2[0]} {p2[1]} {p2[2]} {r} {g} {b}\n')
            f.write(f'v {p3[0]} {p3[1]} {p3[2]} {r} {g} {b}\n')
            f.write(f'v {p4[0]} {p4[1]} {p4[2]} {r} {g} {b}\n')

            # emit a quad face using the four newly written vertices
            f.write(f'f {vertex_counter} {vertex_counter+1} {vertex_counter+2} {vertex_counter+3}\n')
            vertex_counter += 4



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pair',
                        help='Path to a single side-by-side image containing left|right views, or the left image when providing two files')
    parser.add_argument('right', nargs='?', help='Optional right image path. If provided, `pair` is treated as the left image.')
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--out_w', type=int, default=1024)
    parser.add_argument('--out_h', type=int, default=768)
    parser.add_argument('--baseline', type=float, default=0.06)
    parser.add_argument('--num_disp', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=7)
    parser.add_argument('--out', default='cloud.ply')
    parser.add_argument('--debug-save', action='store_true', help='Save left/right perspective crops and disparity visualization')
    parser.add_argument('--projection', choices=['rectilinear', 'spherical'], default='rectilinear', help='Projection mode for equirectangular mapping')
    parser.add_argument('--vfov', type=float, default=None, help='Vertical FOV (degrees). If omitted, derived from aspect ratio')
    parser.add_argument('--input-hfov', type=float, default=360.0, help='Horizontal FOV (degrees) covered by the input equirectangular image (default 360). For VR180 halves use 180.')
    parser.add_argument('--input-vfov', type=float, default=180.0, help='Vertical FOV (degrees) covered by the input equirectangular image (default 180).')
    parser.add_argument('--debug-ply', action='store_true', help='Write a debug PLY where each pixel projects to a point at a fixed radius')
    parser.add_argument('--debug-radius', type=float, default=1.0, help='Radius for debug PLY points (units)')
    args = parser.parse_args()

    # Support either: (A) single side-by-side image, or (B) two separate files
    if args.right:
        left_e = cv2.imread(args.pair, cv2.IMREAD_COLOR)
        right_e = cv2.imread(args.right, cv2.IMREAD_COLOR)
        if left_e is None or right_e is None:
            print('Failed to load one or both images')
            return
    else:
        pair_img = cv2.imread(args.pair, cv2.IMREAD_COLOR)
        if pair_img is None:
            print('Failed to load image:', args.pair)
            return
        h, w = pair_img.shape[:2]
        half = w // 2
        left_e = pair_img[:, :half].copy()
        right_e = pair_img[:, half:half + half].copy()
        # If halves differ by one pixel, crop to the smaller width
        if left_e.shape[1] != right_e.shape[1]:
            minw = min(left_e.shape[1], right_e.shape[1])
            left_e = left_e[:, :minw]
            right_e = right_e[:, :minw]

    left_p, left_vec = equirectangular_to_perspective(left_e, args.fov, args.yaw, args.pitch, args.out_w, args.out_h, projection=args.projection, vfov_deg=args.vfov, input_hfov_deg=args.input_hfov, input_vfov_deg=args.input_vfov)
    right_p, right_vec = equirectangular_to_perspective(right_e, args.fov, args.yaw, args.pitch, args.out_w, args.out_h, projection=args.projection, vfov_deg=args.vfov, input_hfov_deg=args.input_hfov, input_vfov_deg=args.input_vfov)

    left_gray = cv2.cvtColor(left_p, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_p, cv2.COLOR_BGR2GRAY)

    disp = compute_disparity(left_gray, right_gray, num_disp=args.num_disp, block_size=args.block_size)

    # optional debug PLY: map each pixel's direction to a constant-radius point
    if args.debug_ply:
        base = os.path.splitext(args.out)[0]
        # normalize direction vectors and scale to debug radius
        lv = left_vec.astype(np.float64)
        norms = np.linalg.norm(lv, axis=2)
        norms[norms == 0] = 1.0
        unit = (lv / norms[..., None]).reshape(-1, 3)
        pts_debug = unit * float(args.debug_radius)
        # flip Y so +Y points forward in viewer conventions
        pts_debug[:, 1] *= -1.0
        cols = left_p.reshape(-1, 3)
        save_ply(f'{base}_debug.ply', pts_debug, cols)
        print('Saved debug PLY', f'{base}_debug.ply')

    if args.debug_save:
        base = os.path.splitext(args.out)[0]
        cv2.imwrite(f'{base}_left_persp.png', left_p)
        cv2.imwrite(f'{base}_right_persp.png', right_p)

        disp_vis = disp.copy()
        disp_vis[disp_vis < 0] = 0
        maxd = float(disp_vis.max()) if disp_vis.size else 0.0
        if maxd > 0:
            disp_norm = (disp_vis / maxd * 255.0).astype(np.uint8)
        else:
            disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        cv2.imwrite(f'{base}_disp.png', disp_color)

        mask = disp > 0
        left_masked = left_p.copy()
        left_masked[~mask] = (0, 0, 0)
        cv2.imwrite(f'{base}_left_masked.png', left_masked)

    # compute focal length in pixels from horizontal FOV
    f_px = args.out_w / (2.0 * np.tan(np.deg2rad(args.fov) / 2.0))
    pts, colors = disparity_to_pointcloud(disp, left_p, left_vec, right_vec, args.baseline, f_px=f_px)

    save_ply(args.out, pts, colors)
    # also write OBJ with quad faces and per-vertex colors based on the perspective grid
    base = os.path.splitext(args.out)[0]
    # build per-pixel depth grid and vertex grid
    H, W = disp.shape
    dirs = left_vec.astype(np.float64)
    norms = np.linalg.norm(dirs, axis=2)
    norms[norms == 0] = 1.0
    unit_dirs = dirs / norms[..., None]

    Z = np.full((H, W), np.nan, dtype=np.float64)
    valid = disp > 0
    if f_px is not None:
        Z[valid] = (f_px * float(args.baseline)) / disp[valid]

    verts_grid = unit_dirs * Z[..., None]
    cols_grid = left_p.copy()
    save_obj_quads(f'{base}.obj', verts_grid, cols_grid, valid_mask=valid)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
