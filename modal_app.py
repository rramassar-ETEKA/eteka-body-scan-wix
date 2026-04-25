"""
ETEKA Body Scan WIX - Modal Deployment

Fully non-parametric pipeline:
- PIFuHD (Facebook Research) reconstructs a high-detail 3D body mesh from the
  front photo. Non-parametric: captures atypical morphologies (pregnancy, etc.).
- MediaPipe Pose detects 2D/3D keypoints directly on the photo and maps them
  onto the PIFuHD mesh frame. Keypoints align exactly with the visible body.
- Measurements are computed by slicing the PIFuHD mesh at MediaPipe keypoint
  Y levels.
"""

import modal

app = modal.App("eteka-body-scan-wix")

volume = modal.Volume.from_name("size-finder-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev",
        "libglfw3", "libglfw3-dev", "libosmesa6-dev", "freeglut3-dev",
        "libspatialindex-dev",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python",
        "scikit-image",
        "scipy",
        "einops",
        "fastapi",
        "python-multipart",
        "trimesh",
        "shapely",
        "rtree",
        "numpy<2",
        "Pillow",
        "rembg[cpu]",
        "onnxruntime",
        "tqdm",
        "matplotlib",
        "mediapipe==0.10.14",
        "fast-simplification",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/pifuhd.git /opt/pifuhd",
        "mkdir -p /opt/pifuhd/checkpoints && "
        "wget -q https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt "
        "-O /opt/pifuhd/checkpoints/pifuhd.pt",
        # Patch PIFuHD for numpy 2.x / 1.24+ compatibility (np.bool removed)
        "sed -i 's/np.bool)/bool)/g' /opt/pifuhd/lib/sdf.py",
        "sed -i 's/np.bool_/bool/g' /opt/pifuhd/lib/sdf.py",
        "grep -rl 'np.int)' /opt/pifuhd/ 2>/dev/null | xargs -r sed -i 's/np.int)/int)/g'",
        "grep -rl 'np.float)' /opt/pifuhd/ 2>/dev/null | xargs -r sed -i 's/np.float)/float)/g'",
    )
)


# MediaPipe Pose keypoint indices
MP_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=900,
    scaledown_window=120,
)
class BodyScanner:

    @modal.enter()
    def load_models(self):
        import mediapipe as mp
        print("Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        print("MediaPipe ready. PIFuHD loads on first use.")

    def _silhouette_and_bbox(self, image_bytes):
        """Return cropped silhouette + bbox (x1, y1, x2, y2) + image size."""
        from rembg import remove
        from PIL import Image
        import numpy as np
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = remove(img)
        alpha = np.array(out)[:, :, 3]
        sil = (alpha > 128).astype(np.uint8)
        rows = np.any(sil > 0, axis=1)
        cols = np.any(sil > 0, axis=0)
        if not rows.any() or not cols.any():
            return None, None, img.size
        top = rows.argmax()
        bottom = len(rows) - 1 - rows[::-1].argmax()
        left = cols.argmax()
        right = len(cols) - 1 - cols[::-1].argmax()
        return sil[top:bottom + 1, left:right + 1].astype(bool), (left, top, right, bottom), img.size

    def _detect_pose_keypoints(self, image_bytes, bbox, img_size):
        """
        Detect 2D/3D pose keypoints using MediaPipe Pose.
        Returns keypoints mapped to PIFuHD mesh frame:
          - Y: normalized relative to body bbox (0=top/head, 1=bottom/feet)
          - X, Z: normalized relative to body center
        """
        import mediapipe as mp
        from PIL import Image
        import numpy as np
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(img_np)

        if not results.pose_landmarks:
            return None

        W, H = img_size
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        kp_norm = {}
        for name, idx in MP_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            if lm.visibility < 0.3:
                continue
            # Pixel coords in the full image
            px = lm.x * W
            py = lm.y * H
            # Normalize to bbox frame [0,1] (0=top of body, 1=bottom)
            u = (px - x1) / bbox_w
            v = (py - y1) / bbox_h
            # lm.z is relative depth (not reliable), set roughly to 0
            kp_norm[name] = {'u': float(u), 'v': float(v), 'z': float(lm.z)}
        return kp_norm

    def _blur_score(self, image_bytes):
        """Laplacian variance as sharpness proxy. Higher = sharper."""
        import cv2
        import numpy as np
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    def _pose_full(self, image_bytes, model_complexity=1):
        """
        Run MediaPipe Pose, return image-space 2D landmarks + world-space 3D landmarks.
        Returns dict with 'lm_2d' (33 list of dicts {x_px, y_px, vis}),
        'lm_3d' (33 list of dicts {x, y, z}, hip-centered meters),
        'img_size' (W, H), or None if no pose detected.
        """
        import mediapipe as mp
        from PIL import Image
        import numpy as np
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        img_np = np.array(img)

        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(img_np)

        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None

        lm_2d = [
            {"x": float(lm.x * W), "y": float(lm.y * H), "vis": float(lm.visibility)}
            for lm in results.pose_landmarks.landmark
        ]
        lm_3d = [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in results.pose_world_landmarks.landmark
        ]
        return {"lm_2d": lm_2d, "lm_3d": lm_3d, "img_size": (W, H)}

    def _estimate_orientation_deg(self, lm_2d):
        """
        Estimate body rotation around the vertical (Y) axis from 2D landmark VISIBILITY
        scores (much more reliable than MediaPipe world Z which is noisy in profile).

        MediaPipe Pose visibility goes high when a landmark is visible to the camera.
        - At theta=0 (facing camera): nose visible, both eyes/ears visible
        - At theta=90 (subject turned right, camera sees subject's LEFT side): left
          eye/ear visible, right eye/ear hidden
        - At theta=180 (back): no face visible
        - At theta=270 (subject turned left, camera sees subject's RIGHT side): right
          eye/ear visible, left hidden

        Map (front_back, left_right) to angle via atan2.

        MediaPipe landmark indices: 0=nose, 1-3=left eye trio, 4-6=right eye trio,
        7=left ear, 8=right ear.

        Returns angle in [0, 360).
        """
        import math
        nose_v = lm_2d[0]["vis"]
        left_face = (lm_2d[1]["vis"] + lm_2d[2]["vis"] + lm_2d[3]["vis"] + lm_2d[7]["vis"]) / 4.0
        right_face = (lm_2d[4]["vis"] + lm_2d[5]["vis"] + lm_2d[6]["vis"] + lm_2d[8]["vis"]) / 4.0
        side_score = left_face - right_face        # +1 = subject's LEFT side toward camera (theta~90)
        front_back = nose_v * 2.0 - 1.0            # +1 = front, -1 = back

        # atan2(side, front_back):
        #   front (1, 0): atan2(0, 1)  = 0
        #   left-side-visible (0.something, +1): atan2(+1, ~0) = 90
        #   back (-1, 0): atan2(0, -1) = 180 (or -180)
        #   right-side-visible (0.something, -1): atan2(-1, ~0) = -90 = 270
        angle_deg = math.degrees(math.atan2(side_score, front_back))
        if angle_deg < 0:
            angle_deg += 360.0
        return angle_deg

    def _hip_centered_silhouette(self, sil_crop, bbox, hip_px_x):
        """
        Take a silhouette already cropped to its tight bbox and pad it horizontally
        so that hip_px_x (in original full-image coords) lands on the horizontal
        center of the returned silhouette.

        This makes the visual hull rotation axis (= silhouette horizontal center)
        align with the actual subject's vertical axis, even when the camera was
        handheld and not perfectly centered on the subject.
        """
        import numpy as np
        x1, _, x2, _ = bbox
        sil_h, sil_w = sil_crop.shape
        hip_local = float(hip_px_x) - float(x1)
        hip_local = max(0.0, min(float(sil_w), hip_local))
        half_w = int(round(max(hip_local, sil_w - hip_local)))
        if half_w <= 0:
            return sil_crop.astype(bool)
        new_w = 2 * half_w
        new_sil = np.zeros((sil_h, new_w), dtype=bool)
        x0 = half_w - int(round(hip_local))
        x0 = max(0, min(new_w - sil_w, x0))
        new_sil[:, x0:x0 + sil_w] = sil_crop.astype(bool)
        return new_sil

    def _map_kp_to_mesh(self, kp_norm, mesh_bounds, height_cm):
        """
        Map normalized keypoints (u in [0,1] X, v in [0,1] Y top-down) to PIFuHD mesh frame:
          - mesh Y: 0=head, height_m=feet  (matches v)
          - mesh X: centered, range ~[-body_w_m/2, +body_w_m/2]
          - Z: approximate (0 at center)
        """
        target_h = height_cm / 100.0
        # mesh_bounds = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        x_min, y_min, z_min = mesh_bounds[0]
        x_max, y_max, z_max = mesh_bounds[1]
        body_w_m = x_max - x_min
        body_d_m = z_max - z_min

        kp_out = {}
        for name, coord in kp_norm.items():
            u, v = coord['u'], coord['v']
            # v=0 at head (top of body), v=1 at feet (bottom of body)
            mesh_y = v * target_h
            # u=0 at left side of body bbox, u=1 at right side
            mesh_x = x_min + u * body_w_m
            mesh_z = 0.0  # centered; refinement via nearest vertex later
            kp_out[name] = [float(mesh_x), float(mesh_y), float(mesh_z)]
        return kp_out

    def _run_pifuhd(self, image_bytes, bbox, img_size):
        """Run PIFuHD and return the largest-component mesh (in its native frame)."""
        import os, sys, tempfile, shutil, glob, io
        import trimesh
        from PIL import Image

        sys.path.insert(0, "/opt/pifuhd")
        from apps.recon import reconWrapper

        work = tempfile.mkdtemp(prefix="pifuhd_")
        in_dir = os.path.join(work, "input")
        out_dir = os.path.join(work, "output")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            img_path = os.path.join(in_dir, "person.png")
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(img_path)

            # Match PIFuHD Colab rectangle formula: use MediaPipe keypoints
            # radius = 0.65 * max(bbox_width, bbox_height) - gives generous context
            x1, y1, x2, y2 = bbox
            W, H = img_size
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = int(0.65 * max(bw, bh))
            rect_x = cx - radius
            rect_y = cy - radius
            rect_side = 2 * radius
            with open(os.path.join(in_dir, "person_rect.txt"), "w") as f:
                f.write(f"{rect_x} {rect_y} {rect_side} {rect_side}\n")
            print(f"Rect: x={rect_x} y={rect_y} side={rect_side} (img: {W}x{H})")

            cmd = [
                "--dataroot", in_dir, "--results_path", out_dir,
                "--loadSize", "1024", "--resolution", "256",
                "--load_netMR_checkpoint_path", "/opt/pifuhd/checkpoints/pifuhd.pt",
                "--start_id", "-1", "--end_id", "-1",
            ]
            print("Running PIFuHD...")
            try:
                reconWrapper(cmd, use_rect=True)
            except Exception as e:
                print(f"reconWrapper raised: {e}")
                import traceback
                traceback.print_exc()

            obj_files = []
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    if f.endswith(".obj"):
                        obj_files.append(os.path.join(root, f))
            if not obj_files:
                print(f"No PIFuHD output in {out_dir}")
                return None

            mesh = trimesh.load(obj_files[0], process=False)
            print(f"PIFuHD raw: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

            # Keep largest connected component
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    largest = max(components, key=lambda c: len(c.vertices))
                    print(f"{len(components)} components -> kept largest ({len(largest.vertices)} verts)")
                    mesh = largest
            except Exception as e:
                print(f"Component split failed: {e}")

            return mesh
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _fit_mesh_to_visual_hull(self, mesh, silhouettes, height_cm):
        """
        Fit PIFuHD mesh to the visual hull built from all 4 silhouettes.
        For each vertex: cast a ray from body axis (0, y, 0) outward through
        the vertex in XZ plane; find the hull boundary along that ray and
        move the vertex there. The resulting mesh, re-projected to any of
        the 4 views, matches the binary silhouette of that view.
        """
        import numpy as np
        import trimesh

        if not silhouettes or len(silhouettes) < 2:
            return mesh

        body_h_mm = height_cm * 10.0
        body_w_mm = body_h_mm * 0.55
        body_d_mm = body_h_mm * 0.40
        voxel_size_mm = 5

        nx = max(int(body_w_mm / voxel_size_mm), 48)
        ny = max(int(body_h_mm / voxel_size_mm), 128)
        nz = max(int(body_d_mm / voxel_size_mm), 48)
        print(f"Visual hull voxel grid: {nx}x{ny}x{nz}")

        x_coords = (np.arange(nx) + 0.5) * voxel_size_mm - body_w_mm / 2
        y_coords = (np.arange(ny) + 0.5) * voxel_size_mm
        z_coords = (np.arange(nz) + 0.5) * voxel_size_mm - body_d_mm / 2

        voxels = np.ones((nx, ny, nz), dtype=bool)

        def crop_sil(sil):
            rows = np.any(sil > 0, axis=1)
            cols = np.any(sil > 0, axis=0)
            if not rows.any() or not cols.any():
                return None
            t = rows.argmax()
            b = len(rows) - 1 - rows[::-1].argmax()
            l = cols.argmax()
            r = len(cols) - 1 - cols[::-1].argmax()
            return sil[t:b + 1, l:r + 1].astype(bool)

        for view, sil_raw in silhouettes.items():
            sil = crop_sil(sil_raw)
            if sil is None:
                continue
            sil_h, sil_w = sil.shape
            row_idx = np.clip((y_coords / body_h_mm * sil_h).astype(int), 0, sil_h - 1)
            if view == "front":
                x_in = x_coords + body_w_mm / 2
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_xy = sil[np.ix_(row_idx, col_idx)].T
                voxels &= mask_xy[:, :, None]
            elif view == "back":
                x_in = -x_coords + body_w_mm / 2
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_xy = sil[np.ix_(row_idx, col_idx)].T
                voxels &= mask_xy[:, :, None]
            elif view == "left":
                z_in = z_coords + body_d_mm / 2
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                # sil[np.ix_(row_idx, col_idx)] shape: (ny, nz). Need (1, ny, nz).
                mask_yz = sil[np.ix_(row_idx, col_idx)]
                voxels &= mask_yz[None, :, :]
            elif view == "right":
                z_in = -z_coords + body_d_mm / 2
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_yz = sil[np.ix_(row_idx, col_idx)]
                voxels &= mask_yz[None, :, :]

        filled = int(voxels.sum())
        print(f"Visual hull: {filled} filled voxels ({100*filled/voxels.size:.1f}%)")
        if filled < 500:
            print("Visual hull too small, skipping fit")
            return mesh

        # For each vertex: cast ray from (0, y, 0) outward through vertex in XZ plane.
        # Find the last filled voxel along ray = hull boundary.
        verts = mesh.vertices.astype(np.float32)
        verts_mm = verts * 1000.0

        # Direction in XZ for each vertex
        xz = verts_mm[:, [0, 2]]
        dists = np.sqrt((xz ** 2).sum(axis=1))
        # Avoid division by zero (vertices on the central axis)
        safe = dists > 0.5
        dirs = np.zeros_like(xz)
        dirs[safe] = xz[safe] / dists[safe, None]

        # Y voxel index (clamped)
        iy = np.clip((verts_mm[:, 1] / voxel_size_mm).astype(int), 0, ny - 1)

        # Walk rays from t=0 to t=max_search, step=voxel_size/2
        max_search = max(body_w_mm, body_d_mm) / 2 * 1.1
        steps = np.arange(1, max_search / (voxel_size_mm * 0.5), dtype=np.float32) * (voxel_size_mm * 0.5)

        # Initialize last inside distance per vertex
        last_inside = np.zeros(len(verts), dtype=np.float32)

        # Vectorized ray march
        for t in steps:
            px = t * dirs[:, 0]
            pz = t * dirs[:, 1]
            ix = ((px + body_w_mm / 2) / voxel_size_mm).astype(int)
            iz = ((pz + body_d_mm / 2) / voxel_size_mm).astype(int)
            valid = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz) & safe
            filled_mask = np.zeros(len(verts), dtype=bool)
            if valid.any():
                filled_mask[valid] = voxels[ix[valid], iy[valid], iz[valid]]
            # Update last_inside where current ray point is filled
            last_inside = np.where(filled_mask, t, last_inside)

        # Fallback for vertices that never found a hull boundary: keep original
        no_hit = last_inside < 1.0
        last_inside[no_hit] = dists[no_hit]

        # Apply: vertex position = last_inside * direction (XZ). Y unchanged.
        verts_new = verts.copy()
        verts_new[:, 0] = (last_inside * dirs[:, 0]) / 1000.0
        verts_new[:, 2] = (last_inside * dirs[:, 1]) / 1000.0

        # Smooth the mesh to avoid voxel quantization artifacts
        new_mesh = trimesh.Trimesh(vertices=verts_new, faces=mesh.faces, process=False)
        try:
            trimesh.smoothing.filter_taubin(new_mesh, lamb=0.5, nu=-0.53, iterations=5)
        except Exception:
            pass
        return new_mesh

    def _correct_mesh_depth_with_silhouette(self, mesh, side_sil, height_cm,
                                             profile_img_bytes=None):
        """
        SMOOTH deformation of mesh Z to match side silhouette.
        Optionally uses MediaPipe on profile photo for accurate body height calibration
        (silhouette bbox may include hair/shadows, inflating height and underestimating depth).
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        import trimesh

        if side_sil is None:
            return mesh

        rows = np.any(side_sil > 0, axis=1)
        if not rows.any():
            return mesh
        sil_top = rows.argmax()
        sil_bot = len(rows) - 1 - rows[::-1].argmax()
        sil_h = sil_bot - sil_top + 1

        # Try to get accurate body height from MediaPipe Pose on profile photo
        # (sil_h from rembg bbox is often inflated by hair/shadows)
        calibration_factor = 1.0
        if profile_img_bytes is not None:
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(profile_img_bytes)).convert("RGB")
                img_np = np.array(img)
                with self.mp_pose.Pose(
                    static_image_mode=True, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5,
                ) as pose:
                    results = pose.process(img_np)
                if results.pose_landmarks:
                    H_img = img_np.shape[0]
                    nose = results.pose_landmarks.landmark[0]
                    # ankles idx 27 and 28
                    l_ankle = results.pose_landmarks.landmark[27]
                    r_ankle = results.pose_landmarks.landmark[28]
                    ankle_y = max(l_ankle.y, r_ankle.y) * H_img
                    nose_y = nose.y * H_img
                    real_body_height_pix = ankle_y - nose_y
                    # ratio to sil_h
                    if real_body_height_pix > 50:
                        calibration_factor = real_body_height_pix / sil_h
                        print(f"Profile calib: sil_h={sil_h}px, "
                              f"real_body_h={real_body_height_pix:.0f}px, "
                              f"factor={calibration_factor:.3f}")
            except Exception as e:
                print(f"Profile calibration failed: {e}")

        verts = mesh.vertices.astype(np.float32).copy()
        target_h = height_cm / 100.0

        n_levels = 100
        y_samples = np.linspace(0, target_h, n_levels)
        z_extra = np.zeros(n_levels, dtype=np.float32)

        band_width = target_h / n_levels * 2.0
        # Debug: log a few representative levels
        debug_levels = [int(n_levels * f) for f in (0.3, 0.5, 0.7, 0.85)]
        for i in range(n_levels):
            y = y_samples[i]
            mask = np.abs(verts[:, 1] - y) < band_width
            if mask.sum() < 5:
                continue
            mesh_z_extent = float(verts[mask, 2].max() - verts[mask, 2].min())

            y_norm = y / target_h
            row = int(sil_top + y_norm * sil_h)
            row = max(0, min(side_sil.shape[0] - 1, row))
            cols = np.where(side_sil[row] > 0)[0]
            if len(cols) < 2:
                continue
            # Apply calibration: if bbox was inflated (factor<1), depths were understated
            sil_depth_m = ((cols.max() - cols.min()) / sil_h) * target_h / max(calibration_factor, 0.3)

            if i in debug_levels:
                print(f"  Y={y:.2f}m mesh_z={mesh_z_extent*100:.1f}cm "
                      f"sil_z={sil_depth_m*100:.1f}cm")

            # More aggressive correction: smaller margin, bigger cap
            if sil_depth_m > mesh_z_extent + 0.005:  # 5mm margin
                z_extra[i] = min(sil_depth_m - mesh_z_extent, 0.20)  # cap 20cm

        # Moderate smoothing (preserves belly peak better)
        z_extra = gaussian_filter1d(z_extra, sigma=3.0)

        n_affected = (z_extra > 0.01).sum()
        print(f"Belly deformation: {n_affected}/{n_levels} levels "
              f"(max extra={z_extra.max()*100:.1f}cm, smooth front push)")

        if z_extra.max() < 0.01:
            return mesh  # nothing to do

        # Per-vertex extra Z from Y interpolation (continuous)
        y_norm_per_vert = np.clip(verts[:, 1] / target_h, 0, 0.999)
        level_f = y_norm_per_vert * (n_levels - 1)
        level_lo = np.clip(np.floor(level_f).astype(int), 0, n_levels - 1)
        level_hi = np.clip(level_lo + 1, 0, n_levels - 1)
        t = np.clip(level_f - level_lo, 0, 1)
        per_vert_extra = z_extra[level_lo] * (1 - t) + z_extra[level_hi] * t

        # Front weight: 1 at z_max, 0 at z_center, 0 at z_min
        # Cosine easing for smoothness: 0 at center, 1 at front
        z_center = verts[:, 2].mean()
        z_max = verts[:, 2].max()
        front_range = max(z_max - z_center, 1e-4)
        rel = np.clip((verts[:, 2] - z_center) / front_range, 0, 1)
        front_weight = 0.5 - 0.5 * np.cos(np.pi * rel)  # smooth 0->1 cosine

        verts[:, 2] += per_vert_extra * front_weight

        out = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

        # Final smoothing to remove any geometric artifacts from deformation
        try:
            trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=15)
        except Exception:
            pass
        return out

    def _normalize_mesh(self, mesh, height_cm, flip_y=None):
        """Scale + translate: head at Y=0, feet at Y=height_m, centered in X/Z.

        flip_y: bool or None. If None, auto-detect via narrow-end heuristic.
                If True, flip Y. If False, no flip.
        """
        import numpy as np
        import trimesh

        v = mesh.vertices.astype(np.float32)
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        h = y_max - y_min
        if h < 1e-6:
            return mesh

        # Auto-detect flip if not provided: compare width of top 10% vs bottom 10%.
        # Head region has narrow width (neck+head) vs legs region (two legs spread).
        # The NARROWER END is the head.
        if flip_y is None:
            top10 = v[:, 1] < y_min + h * 0.1
            bot10 = v[:, 1] > y_max - h * 0.1
            top_w = (v[top10, 0].max() - v[top10, 0].min()) if top10.sum() > 5 else 0
            bot_w = (v[bot10, 0].max() - v[bot10, 0].min()) if bot10.sum() > 5 else 0
            # If bottom is narrower (= head at bottom in PIFuHD frame), flip
            # Otherwise head is already at top
            flip_y = bot_w < top_w
            print(f"Auto-flip: top_w={top_w:.3f} bot_w={bot_w:.3f} flip={flip_y}")

        target_h = height_cm / 100.0
        scale = target_h / h
        x_mean = v[:, 0].mean()
        z_mean = v[:, 2].mean()

        v_new = v.copy()
        if flip_y:
            v_new[:, 1] = y_max + y_min - v[:, 1]
            v_new[:, 1] = (v_new[:, 1] - y_min) * scale
        else:
            v_new[:, 1] = (v[:, 1] - y_min) * scale
        v_new[:, 0] = (v[:, 0] - x_mean) * scale
        v_new[:, 2] = (v[:, 2] - z_mean) * scale

        return trimesh.Trimesh(vertices=v_new, faces=mesh.faces, process=False)

    def _snap_keypoints_to_mesh(self, kp, mesh):
        """
        For each keypoint (XYZ in mesh frame), snap Z to the nearest mesh surface
        at that (X, Y) position so keypoints sit on the body, not floating.
        """
        import numpy as np
        v = mesh.vertices
        kp_snapped = {}
        for name, xyz in kp.items():
            x, y, z = xyz
            # Find vertices near this Y (within 3cm) and closest in X
            mask = np.abs(v[:, 1] - y) < 0.03
            if mask.sum() < 3:
                kp_snapped[name] = xyz
                continue
            candidates = v[mask]
            # Closest in X
            dists_x = np.abs(candidates[:, 0] - x)
            idx = int(np.argmin(dists_x))
            kp_snapped[name] = [float(candidates[idx, 0]),
                                float(candidates[idx, 1]),
                                float(candidates[idx, 2])]
        return kp_snapped

    def _silhouette_depth_at_y(self, side_sil, y_norm):
        """
        Get the depth (width of side silhouette) at normalized Y level.
        Side silhouette shows the body's front-to-back extent = Z depth.
        y_norm: 0 = top of body, 1 = bottom.
        Returns depth in meters (assuming body height = silhouette height).
        """
        import numpy as np
        if side_sil is None:
            return None
        h, w = side_sil.shape
        y_pix = int(y_norm * (h - 1))
        y_pix = max(0, min(h - 1, y_pix))
        cols = np.where(side_sil[y_pix] > 0)[0]
        if len(cols) == 0:
            return None
        width_pix = cols.max() - cols.min()
        return width_pix  # in pixels

    def _measurements_from_mesh(self, mesh, kp, height_cm, silhouettes=None):
        """
        Anatomically-driven measurement pipeline:
          1. Scan mesh cross-sections along Y from shoulders to below hips
          2. Extract the TORSO polygon at each Y (closest to X=0, filtered)
          3. Detect anatomical landmarks from width profile:
               - chest = local max in upper torso
               - underbust = local min below chest
               - waist = global min between chest and hips
               - hips = local max below waist (iliac crest)
          4. Measure actual polygon perimeter at those Y levels

        For limbs (biceps, thighs, etc.), use MediaPipe keypoints to locate the
        limb and extract its polygon from the slice.
        """
        import trimesh, numpy as np
        import math
        from shapely.geometry import box as shapely_box

        slices_3d = {}
        measurements = {}
        if mesh is None or len(mesh.vertices) == 0:
            return measurements, slices_3d

        def gk(n): return kp.get(n)
        def dist_kp(n1, n2):
            a, b = gk(n1), gk(n2)
            if a is None or b is None: return 0.0
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        ls, rs = gk('left_shoulder'), gk('right_shoulder')
        lh, rh = gk('left_hip'), gk('right_hip')
        if not all([ls, rs, lh, rh]):
            return measurements, slices_3d
        lk, rk = gk('left_knee'), gk('right_knee')
        la, ra = gk('left_ankle'), gk('right_ankle')
        le, re = gk('left_elbow'), gk('right_elbow')

        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        knee_y = (lk[1] + rk[1]) / 2 if (lk and rk) else hip_y + 0.4
        ankle_y = (la[1] + ra[1]) / 2 if (la and ra) else knee_y + 0.4
        elbow_y = (le[1] + re[1]) / 2 if (le and re) else shoulder_y + 0.3

        mesh_verts = mesh.vertices
        shoulder_halfw = abs(ls[0] - rs[0]) / 2
        hip_halfw = abs(lh[0] - rh[0]) / 2

        # Pre-process side silhouettes for depth correction (pregnancy, etc.)
        target_h = height_cm / 100.0
        side_sil = None
        side_sil_h = None
        side_top_y = 0
        if silhouettes is not None:
            sil_keys = list(silhouettes.keys())
            print(f"Silhouettes available: {sil_keys}")
            # Prefer left profile (or right) - profile views give Z depth
            candidate = silhouettes.get("left")
            source = "left"
            if candidate is None:
                candidate = silhouettes.get("right")
                source = "right"
            if candidate is not None:
                import numpy as np
                rows = np.any(candidate > 0, axis=1)
                if rows.any():
                    side_top_y = rows.argmax()
                    side_bot_y = len(rows) - 1 - rows[::-1].argmax()
                    side_sil_h = side_bot_y - side_top_y + 1
                    side_sil = candidate
                    print(f"Using {source} silhouette: shape={candidate.shape} "
                          f"bbox_h={side_sil_h}px")
            else:
                print("No side silhouette available for depth correction")

        def silhouette_depth_m(y_mesh):
            """Get body depth (Z) in meters at mesh Y level, from side silhouette."""
            if side_sil is None or side_sil_h is None:
                return None
            # Mesh Y: 0=top, target_h=bottom. Normalize.
            y_norm = y_mesh / target_h
            pix = self._silhouette_depth_at_y(side_sil, y_norm)
            if pix is None or side_sil_h <= 0:
                return None
            # Convert pixels to meters using body height calibration
            return (pix / side_sil_h) * target_h

        from shapely.geometry import MultiPoint

        def torso_convex_hull(y, half_width_kp, tolerance=0.012):
            """
            Convex hull of torso vertices at Y level with gap detection.
            Returns (hull, width_x, depth_z, perimeter_m, gap_found).
            gap_found is False if torso is still attached to arms at this Y
            (armpit zone) - such Y levels should be excluded from chest detection.
            """
            mask_y = np.abs(mesh_verts[:, 1] - y) < tolerance
            verts = mesh_verts[mask_y]
            if len(verts) < 10:
                return None

            x_limit = max(half_width_kp * 3.0, 0.25)
            mask_x = np.abs(verts[:, 0]) <= x_limit
            central = verts[mask_x]
            if len(central) < 10:
                central = verts

            x_vals = np.sort(central[:, 0])
            center_idx = int(np.argmin(np.abs(x_vals)))
            x_lo = x_vals[0]
            x_hi = x_vals[-1]
            gap_left = False
            gap_right = False

            for i in range(center_idx, 0, -1):
                if (x_vals[i] - x_vals[i - 1]) > 0.012:
                    x_lo = x_vals[i]
                    gap_left = True
                    break
            for i in range(center_idx, len(x_vals) - 1):
                if (x_vals[i + 1] - x_vals[i]) > 0.012:
                    x_hi = x_vals[i]
                    gap_right = True
                    break

            gap_found = gap_left and gap_right

            mask_cluster = (central[:, 0] >= x_lo) & (central[:, 0] <= x_hi)
            cluster = central[mask_cluster]
            if len(cluster) < 10:
                cluster = central

            pts = cluster[:, [0, 2]]
            try:
                hull = MultiPoint([(p[0], p[1]) for p in pts]).convex_hull
                if hull.geom_type != 'Polygon':
                    return None
            except Exception:
                return None

            bounds = hull.bounds
            width_x = bounds[2] - bounds[0]
            depth_z = bounds[3] - bounds[1]

            # Silhouette-based depth correction (captures pregnancy belly bulge)
            sil_depth = silhouette_depth_m(y)
            if sil_depth is not None and sil_depth > depth_z * 1.05:
                print(f"  DEPTH CORRECTION y={y:.3f} mesh_z={depth_z*100:.1f}cm "
                      f"sil_z={sil_depth*100:.1f}cm scale={sil_depth/max(depth_z,1e-4):.2f}")
                # Side silhouette shows greater depth than mesh -> scale Z to match
                z_scale = sil_depth / max(depth_z, 1e-4)
                z_scale = min(z_scale, 1.8)  # cap at 1.8x to avoid artifacts
                # Rebuild hull with scaled Z
                z_center = (bounds[1] + bounds[3]) / 2
                pts_scaled = pts.copy()
                pts_scaled[:, 1] = z_center + (pts[:, 1] - z_center) * z_scale
                try:
                    hull2 = MultiPoint([(p[0], p[1]) for p in pts_scaled]).convex_hull
                    if hull2.geom_type == 'Polygon':
                        hull = hull2
                        bounds = hull.bounds
                        width_x = bounds[2] - bounds[0]
                        depth_z = bounds[3] - bounds[1]
                except Exception:
                    pass

            perim = hull.exterior.length
            return hull, width_x, depth_z, perim, gap_found

        # --- Step 1: Scan torso widths to detect anatomical Y ---
        y_start = shoulder_y + 0.03  # below neck, above chest
        y_end = hip_y + 0.08  # below hip keypoint (real hip is lower)
        n_scan = 50
        ys = np.linspace(y_start, y_end, n_scan)
        widths = np.zeros(n_scan)
        depths = np.zeros(n_scan)
        perims = np.zeros(n_scan)

        gap_valid = np.zeros(n_scan, dtype=bool)
        for i, y in enumerate(ys):
            alpha = (y - shoulder_y) / max(hip_y - shoulder_y, 1e-6)
            alpha = max(0, min(1, alpha))
            halfw = shoulder_halfw * (1 - alpha) + hip_halfw * alpha
            result = torso_convex_hull(y, halfw)
            if result is not None:
                _, w, d, p, gap = result
                widths[i] = w; depths[i] = d; perims[i] = p
                gap_valid[i] = gap

        from scipy.ndimage import uniform_filter1d
        widths_smooth = uniform_filter1d(widths, size=3)

        # --- Step 2: Detect anatomical Y levels ---
        # Only consider Y levels where gap was detected (arms separated from torso)
        # For chest, skip the upper armpit zone (first 15% of scan)
        chest_start = int(n_scan * 0.15)
        chest_end = int(n_scan * 0.40)

        # Mask: valid widths where gap is found, else 0
        valid_widths = np.where(gap_valid, widths_smooth, 0)

        # Chest: max VALID width in 15-40% of scan
        chest_range = valid_widths[chest_start:chest_end + 1]
        if chest_range.max() > 0:
            chest_idx = int(chest_start + np.argmax(chest_range))
        else:
            # No gap found - chest is near shoulders but arms attached
            chest_idx = int(n_scan * 0.20)

        # Hips: max width in lower 35% of scan
        hips_start = int(n_scan * 0.65)
        hips_idx = int(hips_start + np.argmax(widths_smooth[hips_start:]))

        # Waist: min width between chest and hips (valid only)
        if hips_idx > chest_idx + 3:
            waist_range = np.where(
                gap_valid[chest_idx + 1:hips_idx],
                widths_smooth[chest_idx + 1:hips_idx],
                1e9,
            )
            waist_idx = int(chest_idx + 1 + np.argmin(waist_range))
        else:
            waist_idx = chest_idx + (hips_idx - chest_idx) // 2

        # Underbust: local min between chest and waist
        if waist_idx > chest_idx + 3:
            ub_range = np.where(
                gap_valid[chest_idx + 1:waist_idx],
                widths_smooth[chest_idx + 1:waist_idx],
                1e9,
            )
            underbust_idx = int(chest_idx + 1 + np.argmin(ub_range))
        else:
            underbust_idx = min(chest_idx + 2, waist_idx - 1)

        anatomical_y = {
            'chest': ys[chest_idx],
            'underbust': ys[underbust_idx],
            'waist': ys[waist_idx],
            'hips': ys[hips_idx],
        }
        print(f"Anatomical Y: chest={anatomical_y['chest']:.3f} "
              f"underbust={anatomical_y['underbust']:.3f} "
              f"waist={anatomical_y['waist']:.3f} hips={anatomical_y['hips']:.3f}")
        print(f"Widths(cm): chest={widths[chest_idx]*100:.1f} "
              f"ub={widths[underbust_idx]*100:.1f} "
              f"waist={widths[waist_idx]*100:.1f} hips={widths[hips_idx]*100:.1f}")

        # ----- TORSO measurements: convex hull perimeter at anatomical Y -----
        def torso_measure(y_level, name, alpha):
            """Measure convex hull perimeter of torso at given Y."""
            halfw = shoulder_halfw * (1 - alpha) + hip_halfw * alpha
            result = torso_convex_hull(y_level, halfw)
            if result is None:
                return None
            hull, width_x, depth_z, perim, _ = result
            # Save 3D contour
            try:
                coords_2d = list(hull.exterior.coords)
                coords_3d = [[c[0], y_level, c[1]] for c in coords_2d]
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d}
            except Exception:
                pass
            perim_cm = perim * 100.0
            print(f"[{name}] y={y_level:.3f} w={width_x*100:.1f}cm "
                  f"d={depth_z*100:.1f}cm perim={perim_cm:.1f}cm")
            return perim_cm

        for name in ("chest", "underbust", "waist", "hips"):
            y = anatomical_y[name]
            alpha = (y - shoulder_y) / max(hip_y - shoulder_y, 1e-6)
            alpha = max(0, min(1, alpha))
            c = torso_measure(y, name, alpha)
            if c: measurements[name] = round(c, 1)

        # ----- BELLY: max circumference between waist and hips -----
        # Captures pregnancy belly, visceral fat, etc.
        waist_y = anatomical_y['waist']
        hips_y_a = anatomical_y['hips']
        if hips_y_a > waist_y + 0.03:
            n_belly_scan = 20
            belly_ys = np.linspace(waist_y + 0.02, hips_y_a - 0.02, n_belly_scan)
            best_perim = 0
            best_y = None
            for y in belly_ys:
                alpha = (y - shoulder_y) / max(hip_y - shoulder_y, 1e-6)
                halfw = shoulder_halfw * (1 - alpha) + hip_halfw * alpha
                r = torso_convex_hull(y, halfw)
                if r is not None:
                    _, _, _, p, _ = r
                    if p > best_perim:
                        best_perim = p
                        best_y = y
            if best_y is not None and best_perim > 0:
                # Only report if significantly larger than waist (>5cm more)
                waist_perim = measurements.get('waist', 0) / 100.0
                if best_perim > waist_perim + 0.03:
                    alpha = (best_y - shoulder_y) / max(hip_y - shoulder_y, 1e-6)
                    c = torso_measure(best_y, "belly", alpha)
                    if c: measurements["belly"] = round(c, 1)

        # ----- LIMB helpers -----
        def slice_vertices(y, tolerance=0.015):
            mask = np.abs(mesh_verts[:, 1] - y) < tolerance
            return mesh_verts[mask]

        def ellipse_perimeter(a, b):
            if a + b < 1e-6: return 0.0
            h = ((a - b) / (a + b)) ** 2
            return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

        def build_ellipse_contour(y, a, b, cx=0.0, cz=0.0, n=48):
            coords = []
            for i in range(n + 1):
                theta = 2 * math.pi * i / n
                coords.append([cx + a * math.cos(theta), y, cz + b * math.sin(theta)])
            return coords

        def limb_circumf(y, name, target_x, x_radius=0.08):
            """Limb ellipse using cluster detection around target_x."""
            verts = slice_vertices(y)
            if len(verts) < 10: return None
            mask = np.abs(verts[:, 0] - target_x) <= x_radius
            lv = verts[mask]
            if len(lv) < 5: return None
            # Cluster detection
            x_sorted = np.sort(lv[:, 0])
            gaps = np.where(np.diff(x_sorted) > 0.015)[0]
            if len(gaps) > 0:
                idx_target = np.argmin(np.abs(x_sorted - target_x))
                x_lo = x_sorted[0]; x_hi = x_sorted[-1]
                for g in gaps:
                    if g < idx_target:
                        x_lo = x_sorted[g + 1]
                    elif g >= idx_target:
                        x_hi = x_sorted[g]; break
                m2 = (lv[:, 0] >= x_lo) & (lv[:, 0] <= x_hi)
                if m2.sum() >= 5:
                    lv = lv[m2]
            x_min_p, x_max_p = np.percentile(lv[:, 0], [5, 95])
            z_min_p, z_max_p = np.percentile(lv[:, 2], [5, 95])
            a = (x_max_p - x_min_p) / 2
            b = (z_max_p - z_min_p) / 2
            if a < 0.01 or b < 0.01: return None
            cx = (x_max_p + x_min_p) / 2
            cz = (z_max_p + z_min_p) / 2
            slices_3d[name] = {"y": float(y),
                               "contour": build_ellipse_contour(y, a, b, cx, cz)}
            print(f"[{name}] y={y:.3f} target_x={target_x:.3f} a={a:.3f} b={b:.3f} "
                  f"perim={ellipse_perimeter(a, b)*100:.1f}cm")
            return ellipse_perimeter(a, b) * 100.0

        # ----- LEG measurements -----
        if lk and rk:
            target_x_leg = lk[0]
            c = limb_circumf(hip_y + (knee_y - hip_y) * 0.35, "thigh", target_x_leg)
            if c: measurements["thigh"] = round(c, 1)
            c = limb_circumf(knee_y, "knee", target_x_leg)
            if c: measurements["knee"] = round(c, 1)
            calf_target = la[0] if la else target_x_leg
            c = limb_circumf(knee_y + (ankle_y - knee_y) * 0.30, "calf", calf_target)
            if c: measurements["calf"] = round(c, 1)

        # ----- BICEPS (small radius - biceps ~8cm diameter) -----
        if le and re:
            target_x_arm = ls[0] + (le[0] - ls[0]) * 0.40
            c = limb_circumf(shoulder_y + (elbow_y - shoulder_y) * 0.40,
                             "biceps", target_x_arm, x_radius=0.05)
            if c: measurements["biceps"] = round(c, 1)

        # ----- Shoulder width: use mesh X extent at shoulder level -----
        try:
            mask = np.abs(mesh_verts[:, 1] - shoulder_y) < 0.02
            shoulder_verts = mesh_verts[mask]
            if len(shoulder_verts) > 10:
                # 5-95 percentile of X to exclude outliers
                x_min_p, x_max_p = np.percentile(shoulder_verts[:, 0], [2, 98])
                measurements["shoulder_width"] = round((x_max_p - x_min_p) * 100.0, 1)
        except Exception: pass
        if "shoulder_width" not in measurements or measurements["shoulder_width"] < 25:
            # Fallback: keypoints distance x 1.4 (keypoints are at joints, body is wider)
            kp_dist = dist_kp('left_shoulder', 'right_shoulder')
            measurements["shoulder_width"] = round(kp_dist * 100.0 * 1.3, 1)

        # ----- Lengths from keypoints -----
        la_l = dist_kp('left_shoulder', 'left_elbow') + dist_kp('left_elbow', 'left_wrist')
        ra_l = dist_kp('right_shoulder', 'right_elbow') + dist_kp('right_elbow', 'right_wrist')
        if la_l + ra_l > 0:
            measurements["arm_length"] = round((la_l + ra_l) / 2 * 100.0, 1)
        measurements["inseam"] = round(abs(hip_y - ankle_y) * 100.0, 1)
        measurements["torso_length"] = round(abs(shoulder_y - hip_y) * 100.0, 1)
        return measurements, slices_3d

    def _apply_uv_texture(self, mesh, photos):
        """
        UV texture mapping: per-vertex, pick best view via normal alignment,
        project to that view's image, pack in atlas 2x2.
        Returns (uvs array, atlas PNG base64).
        """
        import numpy as np
        from PIL import Image
        import io, base64, trimesh

        verts = mesh.vertices
        # Compute vertex normals if not present
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(verts):
            mesh.vertex_normals  # trigger computation
        normals = np.array(mesh.vertex_normals)

        # PIFuHD coordinate convention: FRONT of body = -Z direction
        # Camera position for front view is at -Z, so normals pointing -Z are best seen
        cam_from_body = {
            "front": np.array([0.0, 0.0, -1.0]),  # front of body faces -Z
            "back":  np.array([0.0, 0.0, 1.0]),
            "left":  np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
        }

        # Score each vertex for each view = normal dot cam_direction
        n_verts = len(verts)
        scores = np.zeros((n_verts, 4), dtype=np.float32)
        view_order = ["front", "left", "back", "right"]
        for i, view in enumerate(view_order):
            scores[:, i] = normals @ cam_from_body[view]

        best_view_idx = np.argmax(scores, axis=1)

        # Mesh bounds for UV normalization
        v_min = verts.min(axis=0)
        v_max = verts.max(axis=0)
        x_min, y_min, z_min = v_min
        x_max, y_max, z_max = v_max
        x_rng = max(x_max - x_min, 1e-6)
        y_rng = max(y_max - y_min, 1e-6)
        z_rng = max(z_max - z_min, 1e-6)

        # Per-vertex UV in LOCAL view space (u, v in [0,1])
        uvs_local = np.zeros((n_verts, 2), dtype=np.float32)

        # Front view: U = X (left-right), V = flipped Y (head at top)
        mask_front = best_view_idx == 0
        uvs_local[mask_front, 0] = (verts[mask_front, 0] - x_min) / x_rng
        uvs_local[mask_front, 1] = 1.0 - (verts[mask_front, 1] - y_min) / y_rng

        # Left profile: U = Z (forward-back), V = flipped Y
        mask_left = best_view_idx == 1
        uvs_local[mask_left, 0] = (verts[mask_left, 2] - z_min) / z_rng
        uvs_local[mask_left, 1] = 1.0 - (verts[mask_left, 1] - y_min) / y_rng

        # Back view: U = flipped X, V = flipped Y
        mask_back = best_view_idx == 2
        uvs_local[mask_back, 0] = 1.0 - (verts[mask_back, 0] - x_min) / x_rng
        uvs_local[mask_back, 1] = 1.0 - (verts[mask_back, 1] - y_min) / y_rng

        # Right profile: U = flipped Z, V = flipped Y
        mask_right = best_view_idx == 3
        uvs_local[mask_right, 0] = 1.0 - (verts[mask_right, 2] - z_min) / z_rng
        uvs_local[mask_right, 1] = 1.0 - (verts[mask_right, 1] - y_min) / y_rng

        # Atlas: 2x2 grid of cropped body photos (each 512x512 -> 1024x1024 total)
        from rembg import remove
        sub = 512
        atlas = Image.new("RGB", (sub * 2, sub * 2), (32, 32, 32))
        atlas_offsets = {
            "front": (0, 0),
            "left":  (sub, 0),
            "back":  (0, sub),
            "right": (sub, sub),
        }
        for view in view_order:
            if view not in photos:
                continue
            img = Image.open(io.BytesIO(photos[view])).convert("RGB")
            # Find body bbox via rembg and crop tight
            alpha = np.array(remove(img))[:, :, 3]
            mask_b = alpha > 128
            if not mask_b.any():
                img_sq = img.resize((sub, sub), Image.LANCZOS)
                atlas.paste(img_sq, atlas_offsets[view])
                continue
            rows_b = np.any(mask_b, axis=1)
            cols_b = np.any(mask_b, axis=0)
            top_b = rows_b.argmax()
            bot_b = len(rows_b) - 1 - rows_b[::-1].argmax()
            left_b = cols_b.argmax()
            right_b = len(cols_b) - 1 - cols_b[::-1].argmax()
            cropped = img.crop((left_b, top_b, right_b + 1, bot_b + 1))
            # Stretch (don't preserve aspect) so body fills entire quadrant
            # -> UV mapping maps mesh bbox directly to body pixels
            stretched = cropped.resize((sub, sub), Image.LANCZOS)
            atlas.paste(stretched, atlas_offsets[view])

        # Atlas UV quadrants: each view occupies a quarter of [0,1]x[0,1]
        # front: u in [0, 0.5], v in [0.5, 1.0]   (upper-left)
        # left:  u in [0.5, 1.0], v in [0.5, 1.0] (upper-right)
        # back:  u in [0, 0.5], v in [0, 0.5]     (lower-left)
        # right: u in [0.5, 1.0], v in [0, 0.5]   (lower-right)
        # Note: PIL paste puts (0,0) at top-left; GL UV (0,0) is bottom-left.
        # So the image Y is already flipped vs UV Y.
        atlas_quads = {
            0: (0.0, 0.5, 0.5, 1.0),  # front
            1: (0.5, 0.0, 0.5, 1.0),  # left
            2: (0.0, 0.0, 0.5, 0.5),  # back
            3: (0.5, 0.0, 1.0, 0.5),  # right
        }
        # Actually let's compute uv offset per quadrant:
        # front @ (0,0) in image (top-left): UV u=[0, 0.5], v=[0.5, 1.0]
        # left @ (sub, 0) in image: u=[0.5, 1.0], v=[0.5, 1.0]
        # back @ (0, sub) in image: u=[0, 0.5], v=[0, 0.5]
        # right @ (sub, sub) in image: u=[0.5, 1.0], v=[0, 0.5]
        uv_offsets = {
            0: (0.0, 0.5),   # front: u starts at 0, v starts at 0.5
            1: (0.5, 0.5),   # left
            2: (0.0, 0.0),   # back
            3: (0.5, 0.0),   # right
        }

        uvs_atlas = np.zeros((n_verts, 2), dtype=np.float32)
        for v_idx in range(4):
            mask = best_view_idx == v_idx
            u_off, v_off = uv_offsets[v_idx]
            uvs_atlas[mask, 0] = u_off + uvs_local[mask, 0] * 0.5
            uvs_atlas[mask, 1] = v_off + uvs_local[mask, 1] * 0.5

        # Encode atlas to JPEG base64 (smaller than PNG)
        buf = io.BytesIO()
        atlas.save(buf, format="JPEG", quality=85)
        atlas_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(f"UV texture: {n_verts} vertices, atlas {sub*2}x{sub*2}, "
              f"views: f={mask_front.sum()} l={mask_left.sum()} "
              f"b={mask_back.sum()} r={mask_right.sum()}")

        return uvs_atlas.tolist(), atlas_b64

    def _build_visual_hull_mesh(self, silhouettes, height_cm, voxel_size_mm=4):
        """
        Build body mesh from 4 silhouettes via voxel-based visual hull + marching cubes.
        Non-parametric, no ML bias: works for any body morphology (pregnancy, obesity...).
        """
        import numpy as np
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
        import trimesh

        body_h_mm = height_cm * 10.0
        body_w_mm = body_h_mm * 0.55
        body_d_mm = body_h_mm * 0.45

        nx = max(int(body_w_mm / voxel_size_mm), 64)
        ny = max(int(body_h_mm / voxel_size_mm), 256)
        nz = max(int(body_d_mm / voxel_size_mm), 64)
        print(f"Visual hull voxel grid: {nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f}M voxels)")

        x_coords = (np.arange(nx) + 0.5) * voxel_size_mm - body_w_mm / 2
        y_coords = (np.arange(ny) + 0.5) * voxel_size_mm
        z_coords = (np.arange(nz) + 0.5) * voxel_size_mm - body_d_mm / 2

        voxels = np.ones((nx, ny, nz), dtype=bool)

        def crop_sil(sil):
            rows = np.any(sil > 0, axis=1)
            cols = np.any(sil > 0, axis=0)
            if not rows.any() or not cols.any():
                return None
            t = rows.argmax()
            b = len(rows) - 1 - rows[::-1].argmax()
            l = cols.argmax()
            r = len(cols) - 1 - cols[::-1].argmax()
            return sil[t:b + 1, l:r + 1].astype(bool)

        for view, sil_raw in silhouettes.items():
            sil = crop_sil(sil_raw)
            if sil is None:
                continue
            sil_h, sil_w = sil.shape
            row_idx = np.clip((y_coords / body_h_mm * sil_h).astype(int), 0, sil_h - 1)
            if view == "front":
                x_in = x_coords + body_w_mm / 2
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_xy = sil[np.ix_(row_idx, col_idx)].T  # (nx, ny)
                voxels &= mask_xy[:, :, None]
            elif view == "back":
                x_in = -x_coords + body_w_mm / 2
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_xy = sil[np.ix_(row_idx, col_idx)].T
                voxels &= mask_xy[:, :, None]
            elif view == "left":
                z_in = z_coords + body_d_mm / 2
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_yz = sil[np.ix_(row_idx, col_idx)]  # (ny, nz)
                voxels &= mask_yz[None, :, :]
            elif view == "right":
                z_in = -z_coords + body_d_mm / 2
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_yz = sil[np.ix_(row_idx, col_idx)]
                voxels &= mask_yz[None, :, :]

        filled = int(voxels.sum())
        print(f"Hull: {filled} filled voxels ({100*filled/voxels.size:.1f}%)")
        if filled < 1000:
            return None

        # Aggressive smoothing for nice isosurface
        v_smooth = gaussian_filter(voxels.astype(np.float32), sigma=1.2)
        v_padded = np.pad(v_smooth, 1, mode='constant', constant_values=0)

        verts, faces, _, _ = marching_cubes(
            v_padded, level=0.5, spacing=(voxel_size_mm,) * 3
        )

        # Re-center
        verts[:, 0] -= voxel_size_mm + body_w_mm / 2
        verts[:, 1] -= voxel_size_mm
        verts[:, 2] -= voxel_size_mm + body_d_mm / 2
        verts /= 1000.0  # mm -> m

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
        except Exception:
            pass

        print(f"Visual hull mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh

    def _do_analyze(self, photos: dict, height_cm: float):
        """
        Clean pipeline:
        - PIFuHD at resolution 512 on front photo (captures body morphology)
        - UV texture from 4 photos (photorealistic)
        - MediaPipe keypoints + measurements
        No corrections/deformations - trust PIFuHD output.
        """
        import numpy as np

        # Silhouettes for UV atlas + measurements
        print("Extracting silhouettes...")
        all_silhouettes = {}
        front_bbox = None
        front_img_size = None
        for view, img_bytes in photos.items():
            sil, bbox, img_size = self._silhouette_and_bbox(img_bytes)
            if sil is not None:
                all_silhouettes[view] = sil
            if view == "front":
                front_bbox = bbox
                front_img_size = img_size

        if front_bbox is None:
            return {"error": "Personne non detectee sur la photo de face"}

        # PIFuHD on FRONT photo (standard colab pipeline)
        print("Running PIFuHD on front photo...")
        pifu_mesh = self._run_pifuhd(photos["front"], front_bbox, front_img_size)
        if pifu_mesh is None:
            return {"error": "Reconstruction 3D echouee"}

        # Only normalize (scale + flip if needed)
        pifu_mesh = self._normalize_mesh(pifu_mesh, height_cm)

        # Correct mesh depth using LEFT profile silhouette (captures belly/pregnancy)
        # Uses smooth cosine easing, asymmetric forward push, Taubin post-smooth
        side_sil_for_correction = all_silhouettes.get("left")
        profile_photo_for_calib = photos.get("left")
        if side_sil_for_correction is None:
            side_sil_for_correction = all_silhouettes.get("right")
            profile_photo_for_calib = photos.get("right")
        if side_sil_for_correction is not None:
            try:
                pifu_mesh = self._correct_mesh_depth_with_silhouette(
                    pifu_mesh, side_sil_for_correction, height_cm,
                    profile_img_bytes=profile_photo_for_calib,
                )
            except Exception as e:
                print(f"Depth correction failed: {e}")
                import traceback
                traceback.print_exc()

        # Decimate mesh for smaller response size
        n_v_before = len(pifu_mesh.vertices)
        print(f"Pre-decimation: {n_v_before} verts, {len(pifu_mesh.faces)} faces")
        if n_v_before > 60000:
            try:
                import fast_simplification
                import trimesh as _trimesh
                import numpy as _np
                target_faces = 80000
                new_v, new_f = fast_simplification.simplify(
                    _np.asarray(pifu_mesh.vertices, dtype=_np.float32),
                    _np.asarray(pifu_mesh.faces, dtype=_np.uint32),
                    target_count=target_faces,
                )
                pifu_mesh = _trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                print(f"Decimated: {len(pifu_mesh.vertices)} verts, "
                      f"{len(pifu_mesh.faces)} faces")
            except Exception as e:
                print(f"Decimation failed: {e}")
                import traceback
                traceback.print_exc()

        # MediaPipe keypoints
        print("Detecting keypoints (MediaPipe)...")
        kp_norm = self._detect_pose_keypoints(photos["front"], front_bbox, front_img_size)
        if kp_norm is None:
            return {"error": "Keypoints non detectes"}

        mesh_bounds = pifu_mesh.bounds
        kp_on_mesh = self._map_kp_to_mesh(kp_norm, mesh_bounds, height_cm)
        kp_snapped = self._snap_keypoints_to_mesh(kp_on_mesh, pifu_mesh)

        # Measurements
        measurements, slices_3d = self._measurements_from_mesh(
            pifu_mesh, kp_snapped, height_cm, silhouettes=all_silhouettes
        )

        # UV texture mapping
        uvs = None
        texture_b64 = None
        try:
            print("Applying UV texture mapping...")
            uvs, texture_b64 = self._apply_uv_texture(pifu_mesh, photos)
        except Exception as e:
            print(f"UV texture failed: {e}")
            import traceback
            traceback.print_exc()

        return {
            "success": True,
            "measurements": measurements,
            "vertices": pifu_mesh.vertices.tolist(),
            "faces": pifu_mesh.faces.tolist(),
            "keypoints_3d": kp_snapped,
            "slices": slices_3d,
            "uvs": uvs,
            "texture_b64": texture_b64,
            "viz_source": "pifuhd_512_uv",
        }

    @modal.method()
    def analyze_multiview(self, photos: dict, height_cm: float = 170.0):
        return self._do_analyze(photos, height_cm)

    def _select_frames_from_jpegs(self, jpegs, n_frames=16):
        """
        Score candidate JPEGs and assign each a rotation angle [0, 360) using
        a temporally-anchored method:

        1. Per frame: extract blur, MediaPipe shoulder signed dx, hip x.
        2. The signed shoulder distance (left_shoulder.x - right_shoulder.x in
           pixels) varies as ~cos(theta) * shoulder_width:
             - max at theta=0 (front, subject's left arm visible at viewer's right)
             - 0 at profile
             - min (negative) at theta=180 (back, arms swapped in image)
        3. Find the t_idx where shoulder_dx is max (= front anchor) and min
           (= back anchor). These are 180 deg apart in time.
        4. Assume rotation is monotonic in time, so the full rotation period
           T_full = 2 * |t_back - t_front|. Map each frame's t_idx to angle:
                   angle(t) = ((t - t_front) / T_full * 360) mod 360
        5. Bucket frames into n_frames angular bins, keep the sharpest per bin.

        Returns sorted list [(angle_deg, jpeg_bytes, hip_px_x), ...] or {"error": str}.
        """
        if len(jpegs) < 4:
            return {"error": "Pas assez de frames candidates"}

        print(f"Scoring {len(jpegs)} candidates (blur + pose)...")
        scored = []
        for t_idx, jpeg in enumerate(jpegs):
            blur = self._blur_score(jpeg)
            pose = self._pose_full(jpeg, model_complexity=1)
            if pose is None:
                continue
            lm = pose["lm_2d"]
            shoulder_dx = float(lm[11]["x"] - lm[12]["x"])  # signed pixels
            hip_px_x = float((lm[23]["x"] + lm[24]["x"]) / 2.0)
            scored.append({
                "jpeg": jpeg, "blur": blur,
                "shoulder_dx": shoulder_dx,
                "hip_px_x": hip_px_x,
                "t_idx": int(t_idx),
            })

        if len(scored) < 4:
            return {"error": "Pose detectee sur trop peu de frames"}

        blurs = sorted(s["blur"] for s in scored)
        blur_floor = max(blurs[len(blurs) // 4] * 0.6, 30.0)
        scored = [s for s in scored if s["blur"] >= blur_floor]
        if len(scored) < 4:
            return {"error": "Trop peu de frames nettes"}
        print(f"Pose+sharp on {len(scored)}/{len(jpegs)} candidates "
              f"(blur floor={blur_floor:.0f})")

        # Temporal anchor on front (max shoulder_dx) and back (min shoulder_dx)
        front_anchor = max(scored, key=lambda s: s["shoulder_dx"])
        back_anchor = min(scored, key=lambda s: s["shoulder_dx"])
        t_front = front_anchor["t_idx"]
        t_back = back_anchor["t_idx"]
        half_period = abs(t_back - t_front)
        if half_period < 2:
            return {"error": "Rotation insuffisante (front et back trop proches dans le temps)"}
        # Sanity: front should have positive shoulder_dx, back negative
        if front_anchor["shoulder_dx"] <= 0 or back_anchor["shoulder_dx"] >= 0:
            print(f"WARNING: shoulder_dx range [{back_anchor['shoulder_dx']:.0f}..{front_anchor['shoulder_dx']:.0f}] "
                  f"does not span both signs - rotation may be incomplete")

        T_full = 2 * half_period
        print(f"Anchors: t_front={t_front} (shoulder_dx={front_anchor['shoulder_dx']:.0f}px), "
              f"t_back={t_back} (shoulder_dx={back_anchor['shoulder_dx']:.0f}px), "
              f"period={T_full} frame-units")

        for s in scored:
            angle = ((s["t_idx"] - t_front) / T_full) * 360.0
            angle = angle % 360.0
            s["angle"] = angle

        buckets = [[] for _ in range(n_frames)]
        for s in scored:
            bi = int(s["angle"] / 360.0 * n_frames) % n_frames
            buckets[bi].append(s)
        selected = []
        for bucket in buckets:
            if not bucket:
                continue
            best = max(bucket, key=lambda x: x["blur"])
            selected.append((best["angle"], best["jpeg"], best["hip_px_x"]))
        selected.sort(key=lambda x: x[0])

        covered = len(selected)
        angles_str = ",".join(f"{a:.0f}" for a, _, _ in selected)
        print(f"Selected {covered}/{n_frames} frames covering angles: [{angles_str}]")
        if covered < 4:
            return {"error": f"Couverture angulaire insuffisante ({covered} vues)"}
        return selected

    @modal.method()
    def analyze_frames(self, jpegs, height_cm: float = 170.0, n_frames: int = 16):
        """
        Pre-extracted-frames variant: takes JPEG frames already pulled from the
        video by the client (saves bandwidth vs uploading the whole video).
        """
        selected = self._select_frames_from_jpegs(jpegs, n_frames=n_frames)
        if isinstance(selected, dict) and "error" in selected:
            return selected
        return self._do_analyze_video(selected, height_cm)

    @modal.method()
    def analyze_video(self, video_bytes: bytes, height_cm: float = 170.0,
                      n_frames: int = 16, n_candidates: int = 32):
        """
        Video-based multi-view reconstruction with robust angle estimation.
        Extracts n_candidates JPEGs from the video then delegates to the same
        scoring/selection/pipeline as analyze_frames.
        """
        import tempfile, os, subprocess, cv2, numpy as np

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(video_bytes)
            raw = f.name
        mp4 = raw.replace(".webm", ".mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-preset", "ultrafast", "-an", mp4],
                check=True, capture_output=True, timeout=60,
            )
        except Exception:
            mp4 = raw

        try:
            cap = cv2.VideoCapture(mp4)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 4:
                return {"error": "Video trop courte"}

            print(f"Extracting {n_candidates} candidate frames from {total} total...")
            indices = np.linspace(0, total - 1, n_candidates).astype(int)
            jpegs = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpegs.append(buf.tobytes())
            cap.release()

            if len(jpegs) < 4:
                return {"error": "Impossible d'extraire assez de frames"}

            selected = self._select_frames_from_jpegs(jpegs, n_frames=n_frames)
            if isinstance(selected, dict) and "error" in selected:
                return selected
            return self._do_analyze_video(selected, height_cm)
        finally:
            for p in (raw, mp4):
                try: os.unlink(p)
                except: pass

    def _do_analyze_video(self, frames, height_cm, enable_texture: bool = False):
        """
        Process N frames (angle, jpeg_bytes, hip_px_x) from rotating video.
        Each frame is a different angle (0..360 deg).
        Silhouettes are recentered horizontally on the MediaPipe hip x position
        so the visual hull rotation axis matches the actual body axis.

        Texture mapping (~60-90s) is skipped by default - mesh + measurements
        are the priority. Set enable_texture=True to compute UV atlas.
        """
        import numpy as np

        n = len(frames)
        print(f"Analyzing {n} frames (angles {frames[0][0]:.0f}..{frames[-1][0]:.0f} deg)")

        # Extract silhouettes + hip-recenter horizontally
        print("Extracting silhouettes (with hip recentering)...")
        silhouettes = []  # list of (angle, hip_centered_silhouette, bbox, img_size)
        raw_for_measure = []  # (angle, raw_tight_sil) for side silhouette in measurements
        for angle, img_bytes, hip_px_x in frames:
            sil, bbox, img_size = self._silhouette_and_bbox(img_bytes)
            if sil is None:
                continue
            sil_centered = self._hip_centered_silhouette(sil, bbox, hip_px_x)
            silhouettes.append((angle, sil_centered, bbox, img_size))
            raw_for_measure.append((angle, sil))

        if len(silhouettes) < 4:
            return {"error": "Trop peu de silhouettes valides"}

        # Build voxel visual hull from all angles
        print(f"Building visual hull from {len(silhouettes)} angles...")
        hull_mesh = self._build_hull_from_rotation(silhouettes, height_cm)
        if hull_mesh is None:
            return {"error": "Reconstruction visual hull echouee"}

        hull_mesh = self._normalize_mesh(hull_mesh, height_cm, flip_y=False)

        # MediaPipe on the first frame (closest to angle 0 = front)
        first_img = frames[0][1]
        sil0, bbox0, img_size0 = self._silhouette_and_bbox(first_img)
        print("Detecting keypoints (MediaPipe)...")
        kp_norm = self._detect_pose_keypoints(first_img, bbox0, img_size0)
        if kp_norm is None:
            return {"error": "Keypoints non detectes"}

        mesh_bounds = hull_mesh.bounds
        kp_on_mesh = self._map_kp_to_mesh(kp_norm, mesh_bounds, height_cm)
        kp_snapped = self._snap_keypoints_to_mesh(kp_on_mesh, hull_mesh)

        # Measurements: pass front (raw, tight) and the frame closest to 90deg as side
        silhouettes_dict = {"front": raw_for_measure[0][1]}
        # Find the frame with angle closest to 90deg (subject's right turned to camera = sees subject's left side)
        # This profile view captures Z-depth (belly bulge for pregnancy)
        side_idx = min(range(len(raw_for_measure)), key=lambda i: abs(raw_for_measure[i][0] - 90))
        if abs(raw_for_measure[side_idx][0] - 90) < 45:
            silhouettes_dict["left"] = raw_for_measure[side_idx][1]
        measurements, slices_3d = self._measurements_from_mesh(
            hull_mesh, kp_snapped, height_cm, silhouettes=silhouettes_dict
        )

        # Multi-view UV texture mapping (skipped by default - saves ~60-90s)
        uvs = None
        texture_b64 = None
        if enable_texture:
            try:
                print("Applying multi-view UV texture...")
                tex_frames = [(a, j) for a, j, _ in frames]
                uvs, texture_b64 = self._apply_uv_texture_multiview(hull_mesh, tex_frames)
            except Exception as e:
                print(f"UV texture failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping UV texture (enable_texture=False, saves ~60-90s)")

        return {
            "success": True,
            "measurements": measurements,
            "vertices": hull_mesh.vertices.tolist(),
            "faces": hull_mesh.faces.tolist(),
            "keypoints_3d": kp_snapped,
            "slices": slices_3d,
            "uvs": uvs,
            "texture_b64": texture_b64,
            "viz_source": "video_visual_hull",
        }

    def _build_hull_from_rotation(self, silhouettes_with_angle, height_cm,
                                    voxel_size_mm=5):
        """
        Build voxel visual hull from N silhouettes, each at a known angle around
        the vertical (Y) axis. Subject assumed to rotate in place, camera fixed.
        """
        import numpy as np
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
        import trimesh

        body_h_mm = height_cm * 10.0
        body_w_mm = body_h_mm * 0.55
        body_d_mm = body_h_mm * 0.55  # symmetrical X/Z for rotation

        nx = max(int(body_w_mm / voxel_size_mm), 64)
        ny = max(int(body_h_mm / voxel_size_mm), 256)
        nz = max(int(body_d_mm / voxel_size_mm), 64)
        print(f"Voxel grid: {nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f}M voxels)")

        # Voxel center coordinates (mm)
        x_coords = (np.arange(nx) + 0.5) * voxel_size_mm - body_w_mm / 2
        y_coords = (np.arange(ny) + 0.5) * voxel_size_mm
        z_coords = (np.arange(nz) + 0.5) * voxel_size_mm - body_d_mm / 2

        voxels = np.ones((nx, ny, nz), dtype=bool)

        def crop_sil(sil):
            # Vertical crop only: keep horizontal extent intact so caller-supplied
            # hip-centered padding is preserved (= rotation axis at horizontal center).
            rows = np.any(sil > 0, axis=1)
            if not rows.any():
                return None
            t = rows.argmax()
            b = len(rows) - 1 - rows[::-1].argmax()
            return sil[t:b + 1, :].astype(bool)

        # For each angle, rotate voxel positions and project to silhouette
        for angle_deg, sil_raw, _, _ in silhouettes_with_angle:
            sil = crop_sil(sil_raw)
            if sil is None:
                continue
            sil_h, sil_w = sil.shape
            theta = np.deg2rad(angle_deg)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Row index for each Y (same for all angles)
            row_idx = np.clip((y_coords / body_h_mm * sil_h).astype(int), 0, sil_h - 1)

            # For each voxel at angle=0 (x, y, z), when subject rotates by angle,
            # camera sees position rotated by -angle:
            # x' = x*cos + z*sin (horizontal in image)
            # Sil width represents X' range from -body_w/2 to +body_w/2
            # Vectorize: compute x' for each (x, z) pair
            x_rot = x_coords[:, None] * cos_t + z_coords[None, :] * sin_t  # (nx, nz)
            # Map x_rot to column index
            col_idx = np.clip(
                ((x_rot + body_w_mm / 2) / body_w_mm * sil_w).astype(int),
                0, sil_w - 1
            )  # (nx, nz)

            # mask[i, j, k] = sil[row_idx[j], col_idx[i, k]]
            # Build using advanced indexing
            mask = sil[row_idx[None, :, None], col_idx[:, None, :]]  # (nx, ny, nz)
            voxels &= mask

        filled = int(voxels.sum())
        print(f"Hull: {filled} voxels ({100*filled/voxels.size:.1f}%)")
        if filled < 1000:
            return None

        # Smooth + marching cubes
        v_smooth = gaussian_filter(voxels.astype(np.float32), sigma=1.0)
        v_padded = np.pad(v_smooth, 1, mode='constant', constant_values=0)
        verts, faces, _, _ = marching_cubes(v_padded, level=0.5,
                                            spacing=(voxel_size_mm,) * 3)
        verts[:, 0] -= voxel_size_mm + body_w_mm / 2
        verts[:, 1] -= voxel_size_mm
        verts[:, 2] -= voxel_size_mm + body_d_mm / 2
        verts /= 1000.0  # mm -> m

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
        except Exception:
            pass

        # Decimate for response size
        if len(mesh.vertices) > 60000:
            try:
                import fast_simplification
                new_v, new_f = fast_simplification.simplify(
                    np.asarray(mesh.vertices, dtype=np.float32),
                    np.asarray(mesh.faces, dtype=np.uint32),
                    target_count=80000,
                )
                mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                print(f"Decimated: {len(mesh.vertices)} verts")
            except Exception as e:
                print(f"Decimation failed: {e}")
        print(f"Hull mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh

    def _apply_uv_texture_multiview(self, mesh, frames, atlas_cols=4):
        """
        Atlas multi-view: arrange N frames in a grid.
        For each vertex, pick best view by normal.dot(camera_dir_at_angle).
        """
        import numpy as np, math, io, base64
        from PIL import Image
        from rembg import remove

        verts = mesh.vertices
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(verts):
            mesh.vertex_normals
        normals = np.array(mesh.vertex_normals)
        n_verts = len(verts)
        n_frames = len(frames)

        # Camera direction for each frame (angle around Y axis)
        # At angle=0, camera at +Z looking -Z (normal alignment: -Z best)
        cam_dirs = np.zeros((n_frames, 3), dtype=np.float32)
        for i, (angle_deg, _) in enumerate(frames):
            theta = np.deg2rad(angle_deg)
            cam_dirs[i] = [-np.sin(theta), 0.0, -np.cos(theta)]

        # Score each vertex per view (normal dot cam_dir)
        scores = normals @ cam_dirs.T  # (n_verts, n_frames)
        best = np.argmax(scores, axis=1)

        # Atlas grid: atlas_cols columns, rows = ceil(n_frames / atlas_cols)
        rows_grid = int(np.ceil(n_frames / atlas_cols))
        sub = 512
        atlas_w = sub * atlas_cols
        atlas_h = sub * rows_grid
        atlas = Image.new("RGB", (atlas_w, atlas_h), (32, 32, 32))

        # Crop body bbox from each frame and paste in atlas
        atlas_offsets = []  # (x_off, y_off) in atlas pixels, (u_off, v_off) in UV
        for i, (angle_deg, img_bytes) in enumerate(frames):
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            alpha = np.array(remove(img))[:, :, 3]
            mask_b = alpha > 128
            if not mask_b.any():
                cropped = img
            else:
                rows_b = np.any(mask_b, axis=1)
                cols_b = np.any(mask_b, axis=0)
                t = rows_b.argmax()
                b = len(rows_b) - 1 - rows_b[::-1].argmax()
                l = cols_b.argmax()
                r = len(cols_b) - 1 - cols_b[::-1].argmax()
                cropped = img.crop((l, t, r + 1, b + 1))
            stretched = cropped.resize((sub, sub), Image.LANCZOS)
            col = i % atlas_cols
            row = i // atlas_cols
            atlas.paste(stretched, (col * sub, row * sub))
            # UV offset (GL convention: V=0 bottom). Image row 0 = top = V at max.
            u_off = col / atlas_cols
            # In image, row 0 is top, but with flipY=false, V=0 is also top
            v_off = row / rows_grid
            atlas_offsets.append((u_off, v_off))

        # Mesh bounds for UV normalization
        v_min = verts.min(axis=0)
        v_max = verts.max(axis=0)
        x_range = max(v_max[0] - v_min[0], 1e-6)
        y_range = max(v_max[1] - v_min[1], 1e-6)
        z_range = max(v_max[2] - v_min[2], 1e-6)

        # Project each vertex to its best view's image plane
        # View at angle θ sees X' = x*cos - z*sin (horizontal in rotated frame)
        uvs = np.zeros((n_verts, 2), dtype=np.float32)
        cell_u = 1.0 / atlas_cols
        cell_v = 1.0 / rows_grid

        for v_idx in range(n_verts):
            k = best[v_idx]
            theta = np.deg2rad(frames[k][0])
            x, y, z = verts[v_idx]
            # Project to rotated frame: camera sees X' horizontal
            x_prime = x * np.cos(theta) + z * np.sin(theta)
            # Body extents along rotated X at this angle - approximate with max(x_range, z_range)
            x_prime_range = max(x_range, z_range)
            u_local = (x_prime - (-x_prime_range / 2)) / x_prime_range
            u_local = max(0.0, min(1.0, u_local))
            v_local = 1.0 - (y - v_min[1]) / y_range
            v_local = max(0.0, min(1.0, v_local))
            u_off, v_off = atlas_offsets[k]
            uvs[v_idx, 0] = u_off + u_local * cell_u
            uvs[v_idx, 1] = v_off + v_local * cell_v

        # Encode atlas
        buf = io.BytesIO()
        atlas.save(buf, format="JPEG", quality=80)
        atlas_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"UV atlas: {atlas_cols}x{rows_grid} grid, {atlas_w}x{atlas_h}px, {n_frames} views")
        return uvs.tolist(), atlas_b64


# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI(title="ETEKA Body Scan WIX API")
web_app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@web_app.get("/")
async def root():
    return {"status": "ok", "pipeline": "pifuhd + mediapipe"}


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


@web_app.post("/analyze_multiview")
async def analyze_multiview(
    photo_front: UploadFile = File(...),
    photo_left: UploadFile = File(...),
    photo_back: UploadFile = File(...),
    photo_right: UploadFile = File(...),
    height_cm: float = Form(170.0),
):
    photos = {
        "front": await photo_front.read(),
        "left": await photo_left.read(),
        "back": await photo_back.read(),
        "right": await photo_right.read(),
    }
    scanner = BodyScanner()
    result = scanner.analyze_multiview.remote(photos, height_cm)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@web_app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...), height_cm: float = Form(170.0)):
    video_bytes = await video.read()
    scanner = BodyScanner()
    result = scanner.analyze_video.remote(video_bytes, height_cm)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@web_app.post("/analyze_video_frames")
async def analyze_video_frames(
    frames: list[UploadFile] = File(...),
    height_cm: float = Form(170.0),
):
    """
    Same pipeline as /analyze_video but receives pre-extracted JPEG frames from
    the client (saves uploading the whole video, ~10x smaller payload).
    """
    if len(frames) < 4:
        raise HTTPException(status_code=400, detail="Au moins 4 frames requises")
    jpegs = [await f.read() for f in frames]
    scanner = BodyScanner()
    result = scanner.analyze_frames.remote(jpegs, height_cm)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.function(image=image, timeout=900)
@modal.asgi_app()
def fastapi_app():
    return web_app
