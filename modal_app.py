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

    def _normalize_mesh(self, mesh, height_cm):
        """Scale + translate: head at Y=0, feet at Y=height_m, centered in X/Z.
        Also detects and flips Y if mesh is upside down.
        """
        import numpy as np
        import trimesh

        v = mesh.vertices.astype(np.float32)
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        h = y_max - y_min
        if h < 1e-6:
            return mesh

        # Check if mesh is upside down (widest part in top half = inverted)
        n_levels = 20
        widths = []
        for i in range(n_levels):
            y0 = y_min + i * h / n_levels
            y1 = y_min + (i + 1) * h / n_levels
            mask = (v[:, 1] >= y0) & (v[:, 1] < y1)
            widths.append(v[mask, 0].max() - v[mask, 0].min() if mask.sum() > 5 else 0)
        widest = int(np.argmax(widths))

        target_h = height_cm / 100.0
        scale = target_h / h
        x_mean = v[:, 0].mean()
        z_mean = v[:, 2].mean()

        v_new = v.copy()
        if widest < n_levels // 2:
            # upside down: flip Y before scaling
            print("Mesh upside down, flipping Y")
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

    def _measurements_from_mesh(self, mesh, kp, height_cm):
        """
        Compute measurements via mesh slicing with POSE-AWARE polygon selection:
        - TORSO: crop slice polygon to X range [-shoulder_half_width*1.3, same]
          so arms are EXCLUDED even when connected to torso
        - BICEPS / THIGH / KNEE / CALF: pick polygon closest to the
          MediaPipe keypoint X for that limb
        """
        import trimesh, numpy as np
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
        lw, rw = gk('left_wrist'), gk('right_wrist')

        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        knee_y = (lk[1] + rk[1]) / 2 if (lk and rk) else hip_y + 0.4
        ankle_y = (la[1] + ra[1]) / 2 if (la and ra) else knee_y + 0.4
        elbow_y = (le[1] + re[1]) / 2 if (le and re) else shoulder_y + 0.3

        # Body half-widths from keypoints for X-based polygon selection
        shoulder_halfw = abs(ls[0] - rs[0]) / 2
        hip_halfw = abs(lh[0] - rh[0]) / 2

        import math
        mesh_verts = mesh.vertices

        def slice_vertices(y, tolerance=0.015):
            """Get mesh vertices close to Y level (within tolerance meters)."""
            mask = np.abs(mesh_verts[:, 1] - y) < tolerance
            return mesh_verts[mask]

        def ellipse_perimeter(a, b):
            """Ramanujan approximation for ellipse perimeter (meters)."""
            if a + b < 1e-6:
                return 0.0
            h = ((a - b) / (a + b)) ** 2
            return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

        def build_ellipse_contour(y, a, b, cx=0.0, cz=0.0, n=48):
            """3D contour points for an ellipse at Y level with semi-axes (a, b)."""
            coords = []
            for i in range(n + 1):
                theta = 2 * math.pi * i / n
                coords.append([cx + a * math.cos(theta), y, cz + b * math.sin(theta)])
            return coords

        def torso_circumf(y, name, half_width, multiplier):
            """
            Torso ellipse from mesh vertices near Y.
            Uses a contiguous-cluster approach around X=0 to exclude arms:
              1. Bin vertices in X direction
              2. Grow cluster outward from X=0 while bins are non-empty
              3. Stop at first gap > 2cm (arm disconnected from torso)
            Uses MediaPipe keypoint half_width * multiplier as hard upper bound.
            """
            verts = slice_vertices(y)
            if len(verts) < 10:
                return None

            x_max_abs = half_width * multiplier
            mask = np.abs(verts[:, 0]) <= x_max_abs
            central = verts[mask]
            if len(central) < 10:
                central = verts

            # Find connected cluster around X=0 via histogram gap detection
            x_sorted = np.sort(central[:, 0])
            # Find biggest gaps on each side of 0
            left = x_sorted[x_sorted <= 0]
            right = x_sorted[x_sorted > 0]
            x_lo, x_hi = x_sorted.min(), x_sorted.max()
            if len(left) > 1:
                left_diffs = np.diff(left)
                # Walk from 0 leftward, stop at first gap >2cm
                for i in range(len(left) - 1, 0, -1):
                    if -left[i - 1] > 0 and (left[i] - left[i - 1]) > 0.02:
                        x_lo = left[i]
                        break
            if len(right) > 1:
                for i in range(len(right) - 1):
                    if (right[i + 1] - right[i]) > 0.02:
                        x_hi = right[i]
                        break

            # Filter to cluster range
            mask = (central[:, 0] >= x_lo) & (central[:, 0] <= x_hi)
            cluster = central[mask]
            if len(cluster) < 10:
                cluster = central

            # Use 5-95 percentile to avoid mesh noise
            x_min_p, x_max_p = np.percentile(cluster[:, 0], [5, 95])
            z_min_p, z_max_p = np.percentile(cluster[:, 2], [5, 95])
            a = (x_max_p - x_min_p) / 2
            b = (z_max_p - z_min_p) / 2
            if a < 0.02 or b < 0.02:
                return None
            cx = (x_max_p + x_min_p) / 2
            cz = (z_max_p + z_min_p) / 2
            slices_3d[name] = {"y": float(y),
                               "contour": build_ellipse_contour(y, a, b, cx, cz)}
            print(f"[{name}] y={y:.3f} kp_halfw={half_width:.3f} mult={multiplier} "
                  f"cluster=[{x_lo:.3f},{x_hi:.3f}] a={a:.3f} b={b:.3f} "
                  f"perim={ellipse_perimeter(a, b)*100:.1f}cm")
            return ellipse_perimeter(a, b) * 100.0

        def limb_circumf(y, name, target_x, x_radius=0.08):
            """
            Limb ellipse from mesh vertices near Y, around target_x.
            Uses cluster detection to isolate the limb from nearby torso/arm.
            """
            verts = slice_vertices(y)
            if len(verts) < 10:
                return None
            # Initial filter around target_x
            mask = np.abs(verts[:, 0] - target_x) <= x_radius
            lv = verts[mask]
            if len(lv) < 5:
                return None

            # Cluster detection: keep only contiguous vertices around target_x
            x_sorted = np.sort(lv[:, 0])
            # Find gaps, keep cluster containing target_x
            gaps = np.where(np.diff(x_sorted) > 0.015)[0]
            if len(gaps) > 0:
                # Walk from target_x outward, stop at first gap
                idx_target = np.argmin(np.abs(x_sorted - target_x))
                x_lo = x_sorted[0]
                x_hi = x_sorted[-1]
                for g in gaps:
                    if g < idx_target:
                        x_lo = x_sorted[g + 1]
                    elif g >= idx_target:
                        x_hi = x_sorted[g]
                        break
                m2 = (lv[:, 0] >= x_lo) & (lv[:, 0] <= x_hi)
                if m2.sum() >= 5:
                    lv = lv[m2]

            # Percentile-based dimensions
            x_min_p, x_max_p = np.percentile(lv[:, 0], [5, 95])
            z_min_p, z_max_p = np.percentile(lv[:, 2], [5, 95])
            a = (x_max_p - x_min_p) / 2
            b = (z_max_p - z_min_p) / 2
            if a < 0.01 or b < 0.01:
                return None
            cx = (x_max_p + x_min_p) / 2
            cz = (z_max_p + z_min_p) / 2
            slices_3d[name] = {"y": float(y),
                               "contour": build_ellipse_contour(y, a, b, cx, cz)}
            print(f"[{name}] y={y:.3f} target_x={target_x:.3f} a={a:.3f} b={b:.3f} "
                  f"perim={ellipse_perimeter(a, b)*100:.1f}cm")
            return ellipse_perimeter(a, b) * 100.0

        # ----- TORSO measurements (cluster around X=0) -----
        # Multipliers reflect that MediaPipe keypoints are at body joints (inner),
        # while the actual body silhouette extends further out:
        #   - shoulder keypoints ~= shoulder joint, real shoulder is wider
        #   - hip keypoints ~= hip joint, real hip (iliac crest) is much wider
        torso_levels = [
            ("chest", 0.20, 1.6),       # chest close to shoulders
            ("underbust", 0.35, 1.6),
            ("waist", 0.55, 1.8),        # waist can be narrower than shoulder keypoint
            ("hips", 1.0, 2.5),          # hip keypoints are very inner
        ]
        for name, frac, multiplier in torso_levels:
            y = shoulder_y + (hip_y - shoulder_y) * frac
            half_w = shoulder_halfw + (hip_halfw - shoulder_halfw) * frac
            c = torso_circumf(y, name, half_w, multiplier)
            if c: measurements[name] = round(c, 1)

        # ----- LEG measurements (single leg via MediaPipe knee X) -----
        if lk and rk:
            # Mid-thigh: between hip and knee
            y_thigh = hip_y + (knee_y - hip_y) * 0.35
            # Target X: halfway between hip and knee (average of left + right leg positions)
            leg_x = (lk[0] + rk[0]) / 2  # zero for most people
            # Actually we want ONE leg, not center. Use one side.
            # Use left knee X as target (arbitrary - body symmetric)
            target_x_leg = lk[0]
            c = limb_circumf(y_thigh, "thigh", target_x_leg)
            if c: measurements["thigh"] = round(c, 1)

            c = limb_circumf(knee_y, "knee", target_x_leg)
            if c: measurements["knee"] = round(c, 1)

            y_calf = knee_y + (ankle_y - knee_y) * 0.30
            calf_target = la[0] if la else target_x_leg
            c = limb_circumf(y_calf, "calf", calf_target)
            if c: measurements["calf"] = round(c, 1)

        # ----- BICEPS (single arm via MediaPipe elbow X) -----
        if le and re:
            y_bic = shoulder_y + (elbow_y - shoulder_y) * 0.40
            # Interpolate X between shoulder and elbow for the left arm
            target_x_arm = ls[0] + (le[0] - ls[0]) * 0.40
            c = limb_circumf(y_bic, "biceps", target_x_arm)
            if c: measurements["biceps"] = round(c, 1)

        # ----- Shoulder width (mesh X extent at shoulder) -----
        try:
            s = slice_at(shoulder_y + (hip_y - shoulder_y) * 0.05)
            if s is not None:
                v_s = np.array(s.vertices)
                measurements["shoulder_width"] = round((v_s[:, 0].max() - v_s[:, 0].min()) * 100.0, 1)
        except: pass
        if "shoulder_width" not in measurements:
            measurements["shoulder_width"] = round(dist_kp('left_shoulder', 'right_shoulder') * 100.0, 1)

        # ----- Lengths from keypoints -----
        la_l = dist_kp('left_shoulder', 'left_elbow') + dist_kp('left_elbow', 'left_wrist')
        ra_l = dist_kp('right_shoulder', 'right_elbow') + dist_kp('right_elbow', 'right_wrist')
        if la_l + ra_l > 0:
            measurements["arm_length"] = round((la_l + ra_l) / 2 * 100.0, 1)
        measurements["inseam"] = round(abs(hip_y - ankle_y) * 100.0, 1)
        measurements["torso_length"] = round(abs(shoulder_y - hip_y) * 100.0, 1)
        return measurements, slices_3d

    def _do_analyze(self, photos: dict, height_cm: float):
        """Fully non-parametric pipeline: PIFuHD + MediaPipe."""
        import numpy as np

        # 1. Silhouette + bbox of front photo
        print("Extracting silhouette...")
        sil, bbox, img_size = self._silhouette_and_bbox(photos["front"])
        if bbox is None:
            return {"error": "Personne non detectee sur la photo de face"}

        # 2. PIFuHD reconstruction
        pifu_mesh = self._run_pifuhd(photos["front"], bbox, img_size)
        if pifu_mesh is None:
            return {"error": "Reconstruction 3D echouee"}

        # 3. Normalize: head at Y=0, feet at height_m
        pifu_mesh = self._normalize_mesh(pifu_mesh, height_cm)

        # 4. MediaPipe keypoints on original photo, mapped to mesh frame
        print("Detecting keypoints (MediaPipe)...")
        kp_norm = self._detect_pose_keypoints(photos["front"], bbox, img_size)
        if kp_norm is None:
            return {"error": "Keypoints non detectes"}

        mesh_bounds = pifu_mesh.bounds
        kp_on_mesh = self._map_kp_to_mesh(kp_norm, mesh_bounds, height_cm)

        # Snap keypoints to mesh surface for clean visualization
        kp_snapped = self._snap_keypoints_to_mesh(kp_on_mesh, pifu_mesh)

        # 5. Measurements via slicing
        measurements, slices_3d = self._measurements_from_mesh(pifu_mesh, kp_snapped, height_cm)

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
