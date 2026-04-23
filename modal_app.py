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

            x1, y1, x2, y2 = bbox
            W, H = img_size
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x)
            y2 = min(H, y2 + pad_y)
            w = x2 - x1
            h = y2 - y1
            side = max(w, h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            with open(os.path.join(in_dir, "person_rect.txt"), "w") as f:
                f.write(f"{x1} {y1} {side} {side}\n")

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

    def _correct_mesh_depth_with_silhouette(self, mesh, side_sil, height_cm):
        """
        Deform mesh Z so body depth matches side silhouette at each Y.
        - Smooth interpolation across Y (no staircase)
        - ASYMMETRIC offset: pushes only the FRONT (belly side) forward,
          keeping the back unchanged (correct for pregnancy, abdominal bulge)
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

        verts = mesh.vertices.astype(np.float32)
        target_h = height_cm / 100.0

        # For each Y level: compute the EXTRA depth that needs to be added
        # (we push front forward rather than scaling symmetrically)
        n_levels = 80
        y_edges = np.linspace(0, target_h, n_levels + 1)
        extra_front_depth = np.zeros(n_levels)  # meters to add to front (max Z) side

        for i in range(n_levels):
            y_center = (y_edges[i] + y_edges[i + 1]) / 2
            mask = (verts[:, 1] >= y_edges[i]) & (verts[:, 1] < y_edges[i + 1])
            if mask.sum() < 5:
                continue
            mesh_z_extent = float(verts[mask, 2].max() - verts[mask, 2].min())

            y_norm = y_center / target_h
            y_pix_in_sil = int(sil_top + y_norm * sil_h)
            y_pix_in_sil = max(0, min(side_sil.shape[0] - 1, y_pix_in_sil))
            cols = np.where(side_sil[y_pix_in_sil] > 0)[0]
            if len(cols) < 2:
                continue
            sil_depth_m = ((cols.max() - cols.min()) / sil_h) * target_h

            if mesh_z_extent > 1e-4 and sil_depth_m > mesh_z_extent + 0.005:
                extra = sil_depth_m - mesh_z_extent
                extra_front_depth[i] = min(extra, 0.18)  # cap at 18cm

        # Smooth along Y with Gaussian (wide sigma for no visible bands)
        extra_smooth = gaussian_filter1d(extra_front_depth, sigma=3.0)

        n_corrected = (extra_smooth > 0.01).sum()
        print(f"Mesh depth correction: {n_corrected}/{n_levels} levels "
              f"(max extra={extra_smooth.max()*100:.1f}cm, asymmetric forward push)")

        # Continuous level index for smooth interpolation
        y_norm_per_vert = np.clip(verts[:, 1] / target_h, 0, 0.999)
        level_f = y_norm_per_vert * n_levels - 0.5
        level_lo = np.clip(np.floor(level_f).astype(int), 0, n_levels - 1)
        level_hi = np.clip(level_lo + 1, 0, n_levels - 1)
        t = np.clip(level_f - level_lo, 0, 1)
        per_vert_extra = extra_smooth[level_lo] * (1 - t) + extra_smooth[level_hi] * t

        # Front = positive Z side. Apply gradient: vertices far from back edge
        # get more of the extra depth. Vertices at back edge don't move.
        z_min = verts[:, 2].min()
        z_max = verts[:, 2].max()
        z_range = max(z_max - z_min, 1e-4)
        # Front weight: 0 at z_min (back), 1 at z_max (front), smooth with cosine
        front_weight = np.clip((verts[:, 2] - z_min) / z_range, 0, 1)
        # Soften so only the front third gets pushed
        front_weight = np.where(front_weight > 0.5,
                                (front_weight - 0.5) * 2.0,  # 0 at mid, 1 at max
                                0.0)
        front_weight = np.clip(front_weight, 0, 1) ** 1.5  # ease-in

        v_new = verts.copy()
        v_new[:, 2] = verts[:, 2] + per_vert_extra * front_weight

        return trimesh.Trimesh(vertices=v_new, faces=mesh.faces, process=False)

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

    def _do_analyze(self, photos: dict, height_cm: float):
        """Fully non-parametric pipeline: PIFuHD + MediaPipe + silhouette depth correction."""
        import numpy as np

        # 1. Silhouettes from all 4 views (for depth correction at belly level)
        print("Extracting silhouettes from all views...")
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

        # 2. PIFuHD reconstruction from front photo
        pifu_mesh = self._run_pifuhd(photos["front"], front_bbox, front_img_size)
        if pifu_mesh is None:
            return {"error": "Reconstruction 3D echouee"}

        # 3. Normalize: head at Y=0, feet at height_m
        pifu_mesh = self._normalize_mesh(pifu_mesh, height_cm)

        # 3b. Correct mesh depth using side silhouette (captures pregnancy, etc.)
        side = all_silhouettes.get("left")
        if side is None:
            side = all_silhouettes.get("right")
        if side is not None:
            pifu_mesh = self._correct_mesh_depth_with_silhouette(pifu_mesh, side, height_cm)

        # 4. MediaPipe keypoints on front photo, mapped to mesh frame
        print("Detecting keypoints (MediaPipe)...")
        kp_norm = self._detect_pose_keypoints(photos["front"], front_bbox, front_img_size)
        if kp_norm is None:
            return {"error": "Keypoints non detectes"}

        mesh_bounds = pifu_mesh.bounds
        kp_on_mesh = self._map_kp_to_mesh(kp_norm, mesh_bounds, height_cm)
        kp_snapped = self._snap_keypoints_to_mesh(kp_on_mesh, pifu_mesh)

        # 5. Measurements via slicing (with silhouette-based depth correction)
        measurements, slices_3d = self._measurements_from_mesh(
            pifu_mesh, kp_snapped, height_cm, silhouettes=all_silhouettes
        )

        return {
            "success": True,
            "measurements": measurements,
            "vertices": pifu_mesh.vertices.tolist(),
            "faces": pifu_mesh.faces.tolist(),
            "keypoints_3d": kp_snapped,
            "slices": slices_3d,
            "viz_source": "pifuhd+mediapipe",
        }

    @modal.method()
    def analyze_multiview(self, photos: dict, height_cm: float = 170.0):
        return self._do_analyze(photos, height_cm)

    @modal.method()
    def analyze_video(self, video_bytes: bytes, height_cm: float = 170.0):
        import tempfile, os, subprocess, cv2

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
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n < 4: return {"error": "Video trop courte"}
            indices = [0, n // 4, n // 2, 3 * n // 4]
            views = ["front", "left", "back", "right"]
            photos = {}
            for idx, name in zip(indices, views):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok: continue
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                photos[name] = buf.tobytes()
            cap.release()
            if len(photos) < 4: return {"error": "Impossible d'extraire 4 frames"}
            return self._do_analyze(photos, height_cm)
        finally:
            for p in (raw, mp4):
                try: os.unlink(p)
                except: pass


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


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
