"""
ETEKA Body Scan WIX - Modal Deployment

Pipeline:
- PIFuHD (Facebook Research) reconstructs a high-detail clothed body mesh from
  the front photo. Captures atypical morphologies (pregnancy, etc.) because it's
  a non-parametric implicit function trained on millions of humans.
- SAM 3D Body provides anatomical keypoints (shoulder/hip/knee Y levels).
- Voxel visual hull from 4 silhouettes provides accurate cross-section dimensions.
- Measurements are extracted from the visual hull mesh at SAM3D keypoint Y levels.
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
        "pytorch-lightning==2.6.0",
        "pyrender",
        "opencv-python",
        "yacs",
        "scikit-image",
        "scipy",
        "einops",
        "timm==1.0.22",
        "dill",
        "pandas",
        "rich",
        "hydra-core==1.3.2",
        "hydra-colorlog",
        "pyrootutils",
        "chumpy",
        "networkx==3.2.1",
        "roma",
        "joblib",
        "seaborn",
        "appdirs",
        "loguru",
        "optree",
        "fvcore",
        "pycocotools",
        "huggingface_hub",
        "smplx",
        "webdataset",
        "omegaconf",
        "fastapi",
        "python-multipart",
        "trimesh",
        "shapely",
        "rtree",
        "numpy<2",
        "Pillow",
        "braceexpand",
        "rembg[cpu]",
        "onnxruntime",
        "tqdm",
        "matplotlib",
    )
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        extra_options="--no-build-isolation --no-deps",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam-3d-body",
        "git clone https://github.com/facebookresearch/pifuhd.git /opt/pifuhd",
        # Download PIFuHD checkpoint (~1.5 GB)
        "mkdir -p /opt/pifuhd/checkpoints && "
        "wget -q https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt "
        "-O /opt/pifuhd/checkpoints/pifuhd.pt",
        # Patch PIFuHD for numpy 2.x / 1.24+ compatibility (np.bool removed)
        "sed -i 's/np.bool)/bool)/g' /opt/pifuhd/lib/sdf.py",
        "sed -i 's/np.bool_/bool/g' /opt/pifuhd/lib/sdf.py",
        "grep -rl 'np.int)' /opt/pifuhd/ 2>/dev/null | xargs -r sed -i 's/np.int)/int)/g'",
        "grep -rl 'np.float)' /opt/pifuhd/ 2>/dev/null | xargs -r sed -i 's/np.float)/float)/g'",
        # Verify patches
        "echo '=== sdf.py patch check ===' && grep -n 'np.bool\\|dtype=bool' /opt/pifuhd/lib/sdf.py || true",
    )
)

KEYPOINTS = {
    'nose': 0,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_hip': 9, 'right_hip': 10,
    'left_knee': 11, 'right_knee': 12,
    'left_ankle': 13, 'right_ankle': 14,
    'right_wrist': 41, 'left_wrist': 62,
    'left_acromion': 67, 'right_acromion': 68,
    'neck': 69,
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
        import sys
        sys.path.insert(0, "/opt/sam-3d-body")
        sys.path.insert(0, "/opt/pifuhd")

        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        print("Loading SAM 3D Body model...")
        self.sam_model, self.sam_cfg = load_sam_3d_body(
            checkpoint_path="/checkpoints/sam-3d-body-dinov3/model.ckpt",
            mhr_path="/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        )
        self.sam_estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.sam_model, model_cfg=self.sam_cfg,
            human_detector=None, human_segmentor=None, fov_estimator=None
        )
        print("SAM 3D loaded.")
        print("PIFuHD will be loaded on first use.")

    def _extract_silhouette(self, image_bytes):
        """Return cropped binary silhouette (body bbox)."""
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
            return None, None
        top = rows.argmax()
        bottom = len(rows) - 1 - rows[::-1].argmax()
        left = cols.argmax()
        right = len(cols) - 1 - cols[::-1].argmax()
        return sil[top:bottom + 1, left:right + 1].astype(bool), (top, bottom, left, right)

    def _silhouette_bbox_in_image(self, image_bytes):
        """Get the body bbox in the original image coords."""
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
            return None
        top = rows.argmax()
        bottom = len(rows) - 1 - rows[::-1].argmax()
        left = cols.argmax()
        right = len(cols) - 1 - cols[::-1].argmax()
        return left, top, right, bottom, img.size  # bbox + image size

    def _reconstruct_keypoints(self, image_bytes):
        """SAM3D keypoints (mesh discarded after this)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            path = f.name
        try:
            outputs = self.sam_estimator.process_one_image(path)
            if not outputs:
                return None
            return {
                "vertices": outputs[0]['pred_vertices'],
                "faces": self.sam_estimator.faces,
                "keypoints_3d": outputs[0]['pred_keypoints_3d'],
            }
        finally:
            os.unlink(path)

    def _run_pifuhd(self, image_bytes, bbox_in_image):
        """
        Run PIFuHD on the image to get a clothed body mesh.
        Returns trimesh.Trimesh in arbitrary coords (will rescale later).
        """
        import os, sys, tempfile, shutil, glob
        import trimesh
        from PIL import Image
        import io

        sys.path.insert(0, "/opt/pifuhd")
        from apps.recon import reconWrapper

        work = tempfile.mkdtemp(prefix="pifuhd_")
        in_dir = os.path.join(work, "input")
        out_dir = os.path.join(work, "output")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Save image
            img_path = os.path.join(in_dir, "person.png")
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(img_path)

            # Build rect: PIFuHD format = "x1 y1 w h" (single line)
            x1, y1, x2, y2, (W, H) = bbox_in_image
            # Add small padding around body
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x)
            y2 = min(H, y2 + pad_y)
            w = x2 - x1
            h = y2 - y1
            # PIFuHD expects square crop; expand to square
            side = max(w, h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            with open(os.path.join(in_dir, "person_rect.txt"), "w") as f:
                f.write(f"{x1} {y1} {side} {side}\n")

            # Run PIFuHD reconWrapper
            cmd = [
                "--dataroot", in_dir,
                "--results_path", out_dir,
                "--loadSize", "1024",
                "--resolution", "256",
                "--load_netMR_checkpoint_path", "/opt/pifuhd/checkpoints/pifuhd.pt",
                "--start_id", "-1",
                "--end_id", "-1",
            ]
            print("Running PIFuHD...")
            try:
                reconWrapper(cmd, use_rect=True)
            except Exception as e:
                print(f"reconWrapper raised: {e}")
                import traceback
                traceback.print_exc()

            # List everything that was created
            created_files = []
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    p = os.path.join(root, f)
                    created_files.append(p)
            print(f"PIFuHD produced {len(created_files)} file(s) in {out_dir}:")
            for f in created_files[:20]:
                print(f"  {f}")

            obj_files = [f for f in created_files if f.endswith(".obj")]
            if not obj_files:
                return None

            mesh = trimesh.load(obj_files[0], process=False)
            print(f"PIFuHD raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

            # Keep only the largest connected component (filter out floating
            # fragments like detached hands, face pieces)
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    largest = max(components, key=lambda c: len(c.vertices))
                    print(f"Filtered {len(components)} components -> kept largest "
                          f"({len(largest.vertices)} verts)")
                    mesh = largest
            except Exception as e:
                print(f"Component split failed: {e}")

            return mesh
        except Exception as e:
            print(f"PIFuHD failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _voxel_visual_hull(self, silhouettes, height_cm, voxel_size_mm=8):
        """Voxel visual hull from 4 silhouettes via marching cubes."""
        import numpy as np
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
        import trimesh

        body_h_mm = height_cm * 10.0
        body_w_mm = body_h_mm * 0.55
        body_d_mm = body_h_mm * 0.40

        nx = max(int(body_w_mm / voxel_size_mm), 48)
        ny = max(int(body_h_mm / voxel_size_mm), 128)
        nz = max(int(body_d_mm / voxel_size_mm), 48)

        x_coords = (np.arange(nx) + 0.5) * voxel_size_mm - body_w_mm / 2.0
        y_coords = (np.arange(ny) + 0.5) * voxel_size_mm
        z_coords = (np.arange(nz) + 0.5) * voxel_size_mm - body_d_mm / 2.0

        voxels = np.ones((nx, ny, nz), dtype=bool)

        def carve(view, sil):
            sil_h, sil_w = sil.shape
            row_idx = np.clip((y_coords / body_h_mm * sil_h).astype(int), 0, sil_h - 1)
            if view in ("front", "back"):
                x_in = x_coords + body_w_mm / 2.0 if view == "front" else -x_coords + body_w_mm / 2.0
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_xy = sil[np.ix_(row_idx, col_idx)].T
                voxels[:] &= mask_xy[:, :, None]
            else:
                z_in = z_coords + body_d_mm / 2.0 if view == "left" else -z_coords + body_d_mm / 2.0
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_zy = sil[np.ix_(row_idx, col_idx)]
                voxels[:] &= mask_zy.T[None, :, :]

        for view, sil in silhouettes.items():
            try:
                carve(view, sil)
            except Exception as e:
                print(f"Carve {view} failed: {e}")

        if voxels.sum() < 200:
            return None

        v_smooth = gaussian_filter(voxels.astype(np.float32), sigma=1.0)
        v_padded = np.pad(v_smooth, 1, mode='constant', constant_values=0)
        verts, faces, _, _ = marching_cubes(v_padded, level=0.5,
                                            spacing=(voxel_size_mm,) * 3)
        verts[:, 0] -= voxel_size_mm + body_w_mm / 2.0
        verts[:, 1] -= voxel_size_mm
        verts[:, 2] -= voxel_size_mm + body_d_mm / 2.0
        verts /= 1000.0  # mm -> m

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
        except Exception:
            pass
        return mesh

    def _normalize_mesh(self, mesh, height_cm):
        """Translate + rescale mesh: head at Y=0, feet at Y=height_m, centered in X/Z."""
        import numpy as np
        v = mesh.vertices.astype(np.float32)
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        h = y_max - y_min
        if h < 1e-6:
            return mesh
        target_h = height_cm / 100.0
        scale = target_h / h
        x_mean = v[:, 0].mean()
        z_mean = v[:, 2].mean()
        v_new = v.copy()
        v_new[:, 1] = (v[:, 1] - y_min) * scale
        v_new[:, 0] = (v[:, 0] - x_mean) * scale
        v_new[:, 2] = (v[:, 2] - z_mean) * scale

        # PIFuHD output may be Y-up; detect by checking if original Y_min was at TOP of body
        # Simpler: use heuristic - we want head at top. Check if "wider part" is at higher or lower Y.
        # For our use, force the convention: Y=0 at top.
        import trimesh
        out = trimesh.Trimesh(vertices=v_new, faces=mesh.faces, process=False)
        return out

    def _flip_mesh_if_needed(self, mesh):
        """Detect if mesh is upside-down (head at bottom) and flip Y."""
        import numpy as np
        v = mesh.vertices
        # Find the row with maximum cross-section width vs minimum
        n_levels = 20
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        widths = []
        for i in range(n_levels):
            y0 = y_min + i * (y_max - y_min) / n_levels
            y1 = y_min + (i + 1) * (y_max - y_min) / n_levels
            mask = (v[:, 1] >= y0) & (v[:, 1] < y1)
            if mask.sum() < 5:
                widths.append(0)
            else:
                widths.append(v[mask, 0].max() - v[mask, 0].min())
        widths = np.array(widths)
        # Body widest in lower half (hips). If widest level is in upper half, mesh is upside down.
        widest = int(widths.argmax())
        if widest < n_levels // 2:
            print("Mesh upside down, flipping Y")
            v_new = v.copy()
            v_new[:, 1] = y_max + y_min - v[:, 1]
            import trimesh
            return trimesh.Trimesh(vertices=v_new, faces=mesh.faces, process=False)
        return mesh

    def _align_keypoints(self, keypoints_3d, sam_vertices, height_cm):
        import numpy as np
        sam_y_min = sam_vertices[:, 1].min()
        sam_y_max = sam_vertices[:, 1].max()
        sam_h = sam_y_max - sam_y_min
        target_h = height_cm / 100.0
        kp_aligned = {}
        x_mean = sam_vertices[:, 0].mean()
        z_mean = sam_vertices[:, 2].mean()
        for name, idx in KEYPOINTS.items():
            if idx >= len(keypoints_3d):
                continue
            kp = keypoints_3d[idx]
            new_y = (kp[1] - sam_y_min) / sam_h * target_h
            new_x = (kp[0] - x_mean) / sam_h * target_h
            new_z = (kp[2] - z_mean) / sam_h * target_h
            kp_aligned[name] = [float(new_x), float(new_y), float(new_z)]
        return kp_aligned

    def _calculate_measurements_from_mesh(self, mesh, kp, height_cm):
        import trimesh, numpy as np

        slices_3d = {}
        measurements = {}
        if mesh is None or len(mesh.vertices) == 0:
            return measurements, slices_3d

        def gk(n): return kp.get(n)
        def dist_kp(n1, n2):
            a, b = gk(n1), gk(n2)
            if a is None or b is None: return 0.0
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        ls, rs, lh, rh = gk('left_shoulder'), gk('right_shoulder'), gk('left_hip'), gk('right_hip')
        if not all([ls, rs, lh, rh]):
            return measurements, slices_3d
        lk, rk, la, ra = gk('left_knee'), gk('right_knee'), gk('left_ankle'), gk('right_ankle')
        le, re = gk('left_elbow'), gk('right_elbow')

        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        knee_y = (lk[1] + rk[1]) / 2 if (lk and rk) else hip_y + 0.4
        ankle_y = (la[1] + ra[1]) / 2 if (la and ra) else knee_y + 0.4
        elbow_y = (le[1] + re[1]) / 2 if (le and re) else shoulder_y + 0.3

        def slice_at(y):
            try: return mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            except: return None

        def torso_circ(y, name):
            try:
                s = slice_at(y)
                if s is None: return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full: return None
                best = min(slice_2d.polygons_full,
                          key=lambda p: p.centroid.x**2 + p.centroid.y**2)
                coords_2d = np.array(best.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y), "contour": coords_3d.tolist()}
                return best.exterior.length * 100.0
            except Exception as e:
                print(f"Slice {name}: {e}")
                return None

        def leg_circ(y, name):
            try:
                s = slice_at(y)
                if s is None: return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full: return None
                polys = list(slice_2d.polygons_full)
                if len(polys) == 1:
                    chosen = polys[0]; half = 0.5
                else:
                    sorted_polys = sorted(polys, key=lambda p: p.area, reverse=True)
                    chosen = sorted_polys[0]
                    for p in sorted_polys:
                        if abs(p.centroid.x) > 0.02: chosen = p; break
                    half = 1.0
                coords_2d = np.array(chosen.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y), "contour": coords_3d.tolist()}
                return chosen.exterior.length * 100.0 * half
            except: return None

        def limb_circ(y, name):
            try:
                s = slice_at(y)
                if s is None: return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full: return None
                polys = list(slice_2d.polygons_full)
                if len(polys) == 1:
                    chosen = polys[0]; half = 0.5
                else:
                    chosen = max(polys, key=lambda p: p.centroid.x**2 + p.centroid.y**2); half = 1.0
                coords_2d = np.array(chosen.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y), "contour": coords_3d.tolist()}
                return chosen.exterior.length * 100.0 * half
            except: return None

        for name, frac in [("chest", 0.20), ("underbust", 0.35), ("waist", 0.55), ("hips", 1.0)]:
            y = shoulder_y + (hip_y - shoulder_y) * frac
            c = torso_circ(y, name)
            if c: measurements[name] = round(c, 1)

        for name, frac in [("thigh", 0.35), ("knee", 1.0)]:
            y = hip_y + (knee_y - hip_y) * frac
            c = leg_circ(y, name)
            if c: measurements[name] = round(c, 1)

        c = leg_circ(knee_y + (ankle_y - knee_y) * 0.30, "calf")
        if c: measurements["calf"] = round(c, 1)

        c = limb_circ(shoulder_y + (elbow_y - shoulder_y) * 0.40, "biceps")
        if c: measurements["biceps"] = round(c, 1)

        try:
            s = slice_at(shoulder_y + (hip_y - shoulder_y) * 0.05)
            if s is not None:
                v = np.array(s.vertices)
                measurements["shoulder_width"] = round((v[:, 0].max() - v[:, 0].min()) * 100.0, 1)
        except: pass
        if "shoulder_width" not in measurements:
            measurements["shoulder_width"] = round(dist_kp('left_shoulder', 'right_shoulder') * 100.0, 1)

        la_l = dist_kp('left_shoulder', 'left_elbow') + dist_kp('left_elbow', 'left_wrist')
        ra_l = dist_kp('right_shoulder', 'right_elbow') + dist_kp('right_elbow', 'right_wrist')
        if la_l + ra_l > 0:
            measurements["arm_length"] = round((la_l + ra_l) / 2 * 100.0, 1)
        measurements["inseam"] = round(abs(hip_y - ankle_y) * 100.0, 1)
        measurements["torso_length"] = round(abs(shoulder_y - hip_y) * 100.0, 1)
        return measurements, slices_3d

    def _do_analyze(self, photos: dict, height_cm: float):
        """
        SAM 3D-only pipeline for consistency between mesh, keypoints and measurements.
        SAM 3D normalizes body to T-pose which ensures:
          - torso cross-sections are clean (arms do not pass through them)
          - keypoints are exactly ON the visible mesh
          - slice contours align perfectly with mesh geometry
        """
        import numpy as np
        import trimesh

        print("Running SAM 3D...")
        sam = self._reconstruct_keypoints(photos["front"])
        if sam is None:
            return {"error": "Personne non detectee sur la photo de face"}

        # Rescale mesh + keypoints to common frame: head at Y=0, feet at Y=height_m
        sam_vertices = np.array(sam["vertices"])
        sam_faces = np.array(sam["faces"])
        sam_y_min = sam_vertices[:, 1].min()
        sam_y_max = sam_vertices[:, 1].max()
        sam_h = sam_y_max - sam_y_min
        target_h = height_cm / 100.0
        scale = target_h / sam_h if sam_h > 0 else 1.0
        x_mean = sam_vertices[:, 0].mean()
        z_mean = sam_vertices[:, 2].mean()

        sam_aligned = sam_vertices.copy()
        sam_aligned[:, 1] = (sam_vertices[:, 1] - sam_y_min) * scale
        sam_aligned[:, 0] = (sam_vertices[:, 0] - x_mean) * scale
        sam_aligned[:, 2] = (sam_vertices[:, 2] - z_mean) * scale

        sam_mesh = trimesh.Trimesh(vertices=sam_aligned, faces=sam_faces, process=False)
        kp_aligned = self._align_keypoints(sam["keypoints_3d"], sam_vertices, height_cm)

        # Measurements from SAM3D mesh, with keypoints exactly on mesh
        measurements, slices_3d = self._calculate_measurements_from_mesh(
            sam_mesh, kp_aligned, height_cm
        )

        return {
            "success": True,
            "measurements": measurements,
            "vertices": sam_mesh.vertices.tolist(),
            "faces": sam_mesh.faces.tolist(),
            "keypoints_3d": kp_aligned,
            "slices": slices_3d,
            "viz_source": "sam3d",
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
    return {"status": "ok", "service": "ETEKA Body Scan WIX API", "pipeline": "pifuhd + visual-hull"}


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
