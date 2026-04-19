"""
ETEKA Body Scan WIX - Modal Deployment

Hybrid pipeline:
- SAM 3D Body provides anatomical keypoints (shoulders/hips/knees positions)
- Voxel-based visual hull from 4 silhouettes provides true 3D body shape
  (works on ANY morphology including pregnancy, atypical bodies)
- Mesh slicing on visual hull provides accurate circumferences
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
    )
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        extra_options="--no-build-isolation --no-deps",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam-3d-body",
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
    timeout=600,
    scaledown_window=120,
)
class BodyScanner:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/opt/sam-3d-body")

        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        print("Loading SAM 3D Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path="/checkpoints/sam-3d-body-dinov3/model.ckpt",
            mhr_path="/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None
        )
        print("Model loaded!")

    def _extract_silhouette(self, image_bytes: bytes):
        """Extract binary silhouette + cropped to body bbox."""
        from rembg import remove
        from PIL import Image
        import numpy as np
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = remove(img)
        alpha = np.array(out)[:, :, 3]
        sil = (alpha > 128).astype(np.uint8)

        # Crop to bbox
        rows = np.any(sil > 0, axis=1)
        cols = np.any(sil > 0, axis=0)
        if not rows.any() or not cols.any():
            return None
        top = rows.argmax()
        bottom = len(rows) - 1 - rows[::-1].argmax()
        left = cols.argmax()
        right = len(cols) - 1 - cols[::-1].argmax()
        return sil[top:bottom + 1, left:right + 1].astype(bool)

    def _reconstruct_keypoints(self, image_bytes: bytes):
        """Run SAM 3D Body to get anatomical keypoints (we discard the mesh)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            path = f.name
        try:
            outputs = self.estimator.process_one_image(path)
            if not outputs:
                return None
            return {
                "vertices": outputs[0]['pred_vertices'],
                "faces": self.estimator.faces,
                "keypoints_3d": outputs[0]['pred_keypoints_3d'],
            }
        finally:
            os.unlink(path)

    def _voxel_visual_hull(self, silhouettes, height_cm, voxel_size_mm=8):
        """
        Vectorized voxel-based visual hull (Shape-from-Silhouette).
        Algorithm:
          1) Build 3D voxel grid covering the body bounding box.
          2) For each view, project every voxel to the silhouette and CARVE
             (set voxel to empty) if it falls outside the silhouette.
          3) Apply Gaussian smoothing on the occupancy field.
          4) Extract a mesh via Marching Cubes (Lorensen & Cline 1987).
          5) Taubin smoothing for surface quality.

        Uses orthographic projection assumption (valid when subject is centered
        and >2m from camera, which matches typical home use).

        Returns trimesh.Trimesh with vertices in METERS, head at Y=0.
        """
        import numpy as np
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
        import trimesh

        body_h_mm = height_cm * 10.0
        body_w_mm = body_h_mm * 0.55  # generous bound for arms/hips
        body_d_mm = body_h_mm * 0.40

        nx = max(int(body_w_mm / voxel_size_mm), 48)
        ny = max(int(body_h_mm / voxel_size_mm), 128)
        nz = max(int(body_d_mm / voxel_size_mm), 48)

        # Voxel center coords (in mm), body centered at origin in X/Z, head at Y=0
        x_coords = (np.arange(nx) + 0.5) * voxel_size_mm - body_w_mm / 2.0
        y_coords = (np.arange(ny) + 0.5) * voxel_size_mm
        z_coords = (np.arange(nz) + 0.5) * voxel_size_mm - body_d_mm / 2.0

        voxels = np.ones((nx, ny, nz), dtype=bool)
        print(f"Voxel grid: {nx}x{ny}x{nz} = {nx*ny*nz} voxels @ {voxel_size_mm}mm")

        def carve(view, sil):
            sil_h, sil_w = sil.shape
            # Y row index for each voxel Y (same for all views)
            row_idx = np.clip((y_coords / body_h_mm * sil_h).astype(int), 0, sil_h - 1)

            if view in ("front", "back"):
                # Front: world X axis maps to silhouette column (left to right)
                # Back: subject is rotated 180deg so X is mirrored
                x_in = x_coords + body_w_mm / 2.0 if view == "front" else -x_coords + body_w_mm / 2.0
                col_idx = np.clip((x_in / body_w_mm * sil_w).astype(int), 0, sil_w - 1)
                # mask_xy[i, j] = True iff (col_idx[i], row_idx[j]) inside silhouette
                mask_xy = sil[np.ix_(row_idx, col_idx)].T  # shape (nx, ny)
                voxels[:] &= mask_xy[:, :, None]  # broadcast over z
            else:  # left or right
                z_in = z_coords + body_d_mm / 2.0 if view == "left" else -z_coords + body_d_mm / 2.0
                col_idx = np.clip((z_in / body_d_mm * sil_w).astype(int), 0, sil_w - 1)
                mask_zy = sil[np.ix_(row_idx, col_idx)]  # shape (ny, nz)
                voxels[:] &= mask_zy.T[None, :, :]  # mask_zy.T shape (nz, ny) -> (1, ny, nz)

        for view, sil in silhouettes.items():
            try:
                carve(view, sil)
            except Exception as e:
                print(f"Carve {view} failed: {e}")

        filled = int(voxels.sum())
        print(f"Visual hull: {filled} filled voxels ({100*filled/voxels.size:.1f}%)")
        if filled < 200:
            return None

        # Smooth occupancy field for clean isosurface
        v_smooth = gaussian_filter(voxels.astype(np.float32), sigma=1.0)
        v_padded = np.pad(v_smooth, 1, mode='constant', constant_values=0)

        # Marching Cubes
        verts, faces, _, _ = marching_cubes(
            v_padded, level=0.5,
            spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        )

        # Re-center: subtract padding + grid offset
        verts[:, 0] -= voxel_size_mm + body_w_mm / 2.0
        verts[:, 1] -= voxel_size_mm
        verts[:, 2] -= voxel_size_mm + body_d_mm / 2.0
        verts /= 1000.0  # mm -> meters

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # Taubin smoothing preserves volume better than Laplacian
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
        except Exception:
            pass
        return mesh

    def _align_keypoints_to_visual_hull(self, keypoints_3d, sam3d_vertices, height_cm):
        """
        Convert SAM3D keypoints (in mesh units, head at min_y, feet at max_y) to
        visual hull coords (in meters, head at Y=0, feet at Y=height_cm/100).
        """
        import numpy as np

        sam3d_y_min = sam3d_vertices[:, 1].min()
        sam3d_y_max = sam3d_vertices[:, 1].max()
        sam3d_h = sam3d_y_max - sam3d_y_min
        target_h = height_cm / 100.0

        kp_aligned = {}
        for name, idx in KEYPOINTS.items():
            if idx >= len(keypoints_3d):
                continue
            kp = keypoints_3d[idx]
            new_y = (kp[1] - sam3d_y_min) / sam3d_h * target_h
            # Keep X/Z roughly proportional (assume body width ~0.4 of height)
            new_x = (kp[0] - sam3d_vertices[:, 0].mean()) / sam3d_h * target_h
            new_z = (kp[2] - sam3d_vertices[:, 2].mean()) / sam3d_h * target_h
            kp_aligned[name] = [float(new_x), float(new_y), float(new_z)]
        return kp_aligned

    def _calculate_measurements_from_mesh(self, mesh, kp_aligned, height_cm):
        """
        Compute body measurements from the visual hull mesh.
        Mesh is in meters, head at Y=0, feet at Y=height_cm/100.
        """
        import trimesh
        import numpy as np

        slices_3d = {}
        measurements = {}

        if mesh is None or len(mesh.vertices) == 0:
            return measurements, slices_3d

        def get_kp(name):
            return kp_aligned.get(name)

        def dist_kp(n1, n2):
            a, b = get_kp(n1), get_kp(n2)
            if a is None or b is None:
                return 0.0
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        # Y levels (in meters, mesh frame)
        ls = get_kp('left_shoulder')
        rs = get_kp('right_shoulder')
        lh = get_kp('left_hip')
        rh = get_kp('right_hip')
        lk = get_kp('left_knee')
        rk = get_kp('right_knee')
        la = get_kp('left_ankle')
        ra = get_kp('right_ankle')
        le = get_kp('left_elbow')
        re = get_kp('right_elbow')

        if not all([ls, rs, lh, rh]):
            return measurements, slices_3d

        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        knee_y = (lk[1] + rk[1]) / 2 if (lk and rk) else hip_y + 0.4
        ankle_y = (la[1] + ra[1]) / 2 if (la and ra) else knee_y + 0.4
        elbow_y = (le[1] + re[1]) / 2 if (le and re) else shoulder_y + 0.3

        def slice_at(y_level):
            try:
                return mesh.section(plane_origin=[0, y_level, 0], plane_normal=[0, 1, 0])
            except Exception:
                return None

        def torso_circ_at(y_level, name):
            """Get torso circumference (closest polygon to center)."""
            try:
                s = slice_at(y_level)
                if s is None:
                    return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None
                best = min(slice_2d.polygons_full,
                          key=lambda p: p.centroid.x**2 + p.centroid.y**2)
                # Build 3D contour
                coords_2d = np.array(best.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                # Convert to cm (mesh is in meters)
                circ_m = best.exterior.length
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                return circ_m * 100.0  # m -> cm
            except Exception as e:
                print(f"Slice {name} error: {e}")
                return None

        def leg_circ_at(y_level, name):
            """Get single leg circumference (off-center polygon)."""
            try:
                s = slice_at(y_level)
                if s is None:
                    return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None
                polys = list(slice_2d.polygons_full)
                if len(polys) == 1:
                    chosen = polys[0]
                    half_factor = 0.5  # one polygon = both legs probably
                else:
                    sorted_polys = sorted(polys, key=lambda p: p.area, reverse=True)
                    chosen = sorted_polys[0]
                    for p in sorted_polys:
                        if abs(p.centroid.x) > 0.02:
                            chosen = p
                            break
                    half_factor = 1.0
                coords_2d = np.array(chosen.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                return chosen.exterior.length * 100.0 * half_factor
            except Exception as e:
                print(f"Leg slice {name} error: {e}")
                return None

        def limb_circ_at(y_level, name):
            try:
                s = slice_at(y_level)
                if s is None:
                    return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None
                polys = list(slice_2d.polygons_full)
                if len(polys) == 1:
                    chosen = polys[0]
                    half_factor = 0.5
                else:
                    chosen = max(polys, key=lambda p: p.centroid.x**2 + p.centroid.y**2)
                    half_factor = 1.0
                coords_2d = np.array(chosen.exterior.coords)
                ones = np.ones(len(coords_2d))
                coords_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), ones])
                coords_3d = (to_3d @ coords_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                return chosen.exterior.length * 100.0 * half_factor
            except Exception:
                return None

        # Torso measurements
        torso_levels = [
            ("chest", 0.20),
            ("underbust", 0.35),
            ("waist", 0.55),
            ("hips", 1.0),
        ]
        for name, frac in torso_levels:
            y = shoulder_y + (hip_y - shoulder_y) * frac
            c = torso_circ_at(y, name)
            if c:
                measurements[name] = round(c, 1)

        # Leg measurements
        for name, frac in [("thigh", 0.35), ("knee", 1.0)]:
            y = hip_y + (knee_y - hip_y) * frac
            c = leg_circ_at(y, name)
            if c:
                measurements[name] = round(c, 1)

        y_calf = knee_y + (ankle_y - knee_y) * 0.30
        c = leg_circ_at(y_calf, "calf")
        if c:
            measurements["calf"] = round(c, 1)

        # Biceps
        y_bic = shoulder_y + (elbow_y - shoulder_y) * 0.40
        c = limb_circ_at(y_bic, "biceps")
        if c:
            measurements["biceps"] = round(c, 1)

        # Shoulder width: max X width of mesh near shoulder Y
        shoulder_slice_y = shoulder_y + (hip_y - shoulder_y) * 0.05
        try:
            s = slice_at(shoulder_slice_y)
            if s is not None:
                v = np.array(s.vertices)
                sw = (v[:, 0].max() - v[:, 0].min()) * 100.0
                measurements["shoulder_width"] = round(sw, 1)
        except Exception:
            pass
        if "shoulder_width" not in measurements:
            measurements["shoulder_width"] = round(dist_kp('left_shoulder', 'right_shoulder') * 100.0, 1)

        # Lengths from keypoints
        left_arm = dist_kp('left_shoulder', 'left_elbow') + dist_kp('left_elbow', 'left_wrist')
        right_arm = dist_kp('right_shoulder', 'right_elbow') + dist_kp('right_elbow', 'right_wrist')
        if left_arm + right_arm > 0:
            measurements["arm_length"] = round((left_arm + right_arm) / 2 * 100.0, 1)
        measurements["inseam"] = round(abs(hip_y - ankle_y) * 100.0, 1)
        measurements["torso_length"] = round(abs(shoulder_y - hip_y) * 100.0, 1)

        return measurements, slices_3d

    def _align_sam3d_mesh(self, sam_vertices, height_cm):
        """
        Rescale and translate SAM3D mesh to match visual hull frame:
        - head at Y=0, feet at Y=height_cm/100 (meters)
        - centered at origin in X and Z
        """
        import numpy as np
        v = np.array(sam_vertices, dtype=np.float32)
        y_min = v[:, 1].min()
        y_max = v[:, 1].max()
        h = y_max - y_min
        if h < 1e-6:
            return v
        target_h = height_cm / 100.0
        scale = target_h / h
        x_mean = v[:, 0].mean()
        z_mean = v[:, 2].mean()
        out = v.copy()
        out[:, 1] = (v[:, 1] - y_min) * scale
        out[:, 0] = (v[:, 0] - x_mean) * scale
        out[:, 2] = (v[:, 2] - z_mean) * scale
        return out

    def _do_analyze(self, photos: dict, height_cm: float):
        """
        Hybrid pipeline:
        1) SAM3D mesh -> used for VISUALIZATION (clean body shape)
        2) Voxel visual hull -> used for MEASUREMENTS (true cross-sections)
        3) Both meshes are aligned in the same coord frame so slice contours
           from the visual hull display naturally on the SAM3D mesh.
        """
        import numpy as np

        # 1. SAM3D mesh + keypoints
        base = self._reconstruct_keypoints(photos["front"])
        if base is None:
            return {"error": "Personne non detectee sur la photo de face"}

        # 2. Extract silhouettes
        silhouettes = {}
        for view_name, img_bytes in photos.items():
            sil = self._extract_silhouette(img_bytes)
            if sil is not None:
                silhouettes[view_name] = sil
            else:
                print(f"Silhouette extraction failed for {view_name}")

        if len(silhouettes) < 2:
            return {"error": "Impossible d'extraire les silhouettes"}

        # 3. Voxel visual hull (used for measurements only)
        print(f"Building voxel visual hull from {len(silhouettes)} views...")
        vh_mesh = self._voxel_visual_hull(silhouettes, height_cm)
        if vh_mesh is None:
            return {"error": "Reconstruction visual hull echouee"}
        print(f"Visual hull mesh: {len(vh_mesh.vertices)} verts")

        # 4. Align SAM3D mesh to same frame as visual hull
        sam_aligned = self._align_sam3d_mesh(np.array(base["vertices"]), height_cm)

        # 5. Align SAM3D keypoints to that frame too
        kp_aligned = self._align_keypoints_to_visual_hull(
            base["keypoints_3d"], np.array(base["vertices"]), height_cm
        )

        # 6. Measurements & slice contours from visual hull
        measurements, slices_3d = self._calculate_measurements_from_mesh(
            vh_mesh, kp_aligned, height_cm
        )

        return {
            "success": True,
            "measurements": measurements,
            # SAM3D mesh for visualization (clean body shape)
            "vertices": sam_aligned.tolist(),
            "faces": np.array(base["faces"]).tolist(),
            "keypoints_3d": kp_aligned,
            # Visual hull cross-sections (the "real" measurements)
            "slices": slices_3d,
        }

    @modal.method()
    def analyze_multiview(self, photos: dict, height_cm: float = 170.0):
        return self._do_analyze(photos, height_cm)

    @modal.method()
    def analyze_video(self, video_bytes: bytes, height_cm: float = 170.0,
                      n_frames: int = 8):
        """
        Premium 360 video mode: extract many frames for richer visual hull.
        We extract n_frames evenly spaced and treat them as views around the body.
        For now we map them onto front/left/back/right (pairs averaged).
        """
        import tempfile, os, subprocess
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(video_bytes)
            raw_path = f.name

        mp4_path = raw_path.replace(".webm", ".mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264", "-preset", "ultrafast",
                 "-an", mp4_path],
                check=True, capture_output=True, timeout=60,
            )
        except Exception as e:
            print(f"ffmpeg conversion failed, trying raw: {e}")
            mp4_path = raw_path

        try:
            cap = cv2.VideoCapture(mp4_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 4:
                return {"error": "Video trop courte ou illisible"}

            # Premium: extract 4 cardinal frames; could be expanded to 8/16 with octahedral hull
            indices = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4]
            views = ["front", "left", "back", "right"]
            photos = {}
            for idx, name in zip(indices, views):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                photos[name] = buf.tobytes()
            cap.release()
            if len(photos) < 4:
                return {"error": "Impossible d'extraire 4 frames de la video"}

            return self._do_analyze(photos, height_cm)
        finally:
            for p in [raw_path, mp4_path]:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI(title="ETEKA Body Scan WIX API")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    return {"status": "ok", "service": "ETEKA Body Scan WIX API", "pipeline": "voxel-visual-hull"}


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
async def analyze_video(
    video: UploadFile = File(...),
    height_cm: float = Form(170.0),
):
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
