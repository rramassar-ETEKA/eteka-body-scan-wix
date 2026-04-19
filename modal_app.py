"""
ETEKA Body Scan WIX - Modal Deployment
Multi-view reconstruction: SAM 3D Body (front view) + visual hull refinement
from 4 photos (front/left/back/right) or 360deg video.
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
        "rembg[new]",  # For background removal / silhouettes
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
        """Extract binary silhouette from an image using rembg."""
        from rembg import remove
        from PIL import Image
        import numpy as np
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = remove(img)
        alpha = np.array(out)[:, :, 3]
        silhouette = (alpha > 128).astype(np.uint8)
        return silhouette, img.size

    def _reconstruct_mesh(self, image_bytes: bytes):
        """Run SAM 3D Body on a single image to get base mesh + keypoints."""
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

    def _refine_with_silhouettes(self, base_mesh, silhouettes):
        """
        Refine the mesh shape using silhouettes from 4 views.
        This adjusts per-vertex scaling in X and Z based on view-specific widths.
        """
        import numpy as np

        vertices = np.array(base_mesh["vertices"]).copy()

        # Compute silhouette widths at different Y levels for each view
        # Front/back views give X width, left/right profiles give Z width
        view_widths = {}
        for view_name, sil in silhouettes.items():
            h, w = sil.shape
            # For each Y row, find leftmost and rightmost non-zero
            widths = []
            for y in range(h):
                cols = np.where(sil[y] > 0)[0]
                if len(cols) > 0:
                    widths.append((y / h, (cols.max() - cols.min()) / w))
                else:
                    widths.append((y / h, 0.0))
            view_widths[view_name] = widths

        # Compute mesh dimensions
        mesh_y_min = vertices[:, 1].min()
        mesh_y_max = vertices[:, 1].max()
        mesh_height = mesh_y_max - mesh_y_min

        # Build Y-normalized width tables
        def interp_width(view, y_norm):
            table = view_widths.get(view, [])
            if not table:
                return 1.0
            # Find closest y_norm entries and interpolate
            for i in range(len(table) - 1):
                if table[i][0] <= y_norm <= table[i + 1][0]:
                    return (table[i][1] + table[i + 1][1]) / 2
            return 0.0

        # Compute scale factors per Y level
        # For each mesh vertex, compute Y normalization (inverted since images are top-down)
        refined = vertices.copy()
        mesh_x_center = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
        mesh_z_center = (vertices[:, 2].max() + vertices[:, 2].min()) / 2

        for i in range(len(refined)):
            y_norm = 1.0 - (refined[i, 1] - mesh_y_min) / mesh_height  # inverted

            # Front/back silhouette informs X width
            front_w = interp_width("front", y_norm)
            back_w = interp_width("back", y_norm)
            avg_x_w = (front_w + back_w) / 2 if (front_w > 0 and back_w > 0) else max(front_w, back_w)

            # Left/right silhouette informs Z width
            left_w = interp_width("left", y_norm)
            right_w = interp_width("right", y_norm)
            avg_z_w = (left_w + right_w) / 2 if (left_w > 0 and right_w > 0) else max(left_w, right_w)

            # Compute current mesh widths at this Y level
            same_y_mask = np.abs(vertices[:, 1] - refined[i, 1]) < mesh_height * 0.02
            if same_y_mask.sum() > 3:
                current_x_range = vertices[same_y_mask, 0].max() - vertices[same_y_mask, 0].min()
                current_z_range = vertices[same_y_mask, 2].max() - vertices[same_y_mask, 2].min()

                if avg_x_w > 0.01 and current_x_range > 0:
                    # Normalize by image aspect - use mesh_height/3 as torso reference
                    target_x_range = avg_x_w * mesh_height * 0.6
                    scale_x = target_x_range / current_x_range if current_x_range > 0 else 1.0
                    scale_x = np.clip(scale_x, 0.85, 1.15)  # Limit refinement to +/-15%
                    refined[i, 0] = mesh_x_center + (refined[i, 0] - mesh_x_center) * scale_x

                if avg_z_w > 0.01 and current_z_range > 0:
                    target_z_range = avg_z_w * mesh_height * 0.6
                    scale_z = target_z_range / current_z_range if current_z_range > 0 else 1.0
                    scale_z = np.clip(scale_z, 0.85, 1.15)
                    refined[i, 2] = mesh_z_center + (refined[i, 2] - mesh_z_center) * scale_z

        return refined

    @modal.method()
    def analyze_multiview(self, photos: dict, height_cm: float = 170.0):
        """
        photos: dict with keys 'front', 'left', 'back', 'right' -> bytes
        """
        import numpy as np

        # 1. Reconstruct base mesh from front photo
        base = self._reconstruct_mesh(photos["front"])
        if base is None:
            return {"error": "Personne non detectee sur la photo de face"}

        # 2. Extract silhouettes from all 4 views
        silhouettes = {}
        for view_name, img_bytes in photos.items():
            try:
                sil, _ = self._extract_silhouette(img_bytes)
                silhouettes[view_name] = sil
            except Exception as e:
                print(f"Silhouette extraction failed for {view_name}: {e}")

        # 3. Refine mesh with silhouettes
        if len(silhouettes) >= 2:
            refined_vertices = self._refine_with_silhouettes(base, silhouettes)
        else:
            refined_vertices = np.array(base["vertices"])

        # 4. Compute measurements
        measurements, slices_3d = self._calculate_measurements(
            refined_vertices, base["faces"], base["keypoints_3d"], height_cm
        )

        # 5. Build keypoints dict
        keypoints_labeled = {}
        for name, idx in KEYPOINTS.items():
            if idx < len(base["keypoints_3d"]):
                keypoints_labeled[name] = base["keypoints_3d"][idx].tolist()

        return {
            "success": True,
            "measurements": measurements,
            "vertices": refined_vertices.tolist(),
            "faces": base["faces"].tolist(),
            "keypoints_3d": keypoints_labeled,
            "slices": slices_3d,
        }

    @modal.method()
    def analyze_video(self, video_bytes: bytes, height_cm: float = 170.0):
        """Extract frames from 360 video and run multi-view analysis."""
        import tempfile, os
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 4:
                return {"error": "Video trop courte"}

            # Sample 4 frames evenly spaced for front/left/back/right
            indices = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4]
            views = ["front", "left", "back", "right"]
            photos = {}

            for idx, view_name in zip(indices, views):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                # Encode to JPEG bytes
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                photos[view_name] = buf.tobytes()
            cap.release()

            if len(photos) < 4:
                return {"error": "Impossible d'extraire 4 frames"}

            return self.analyze_multiview.local(photos, height_cm)
        finally:
            os.unlink(video_path)

    def _calculate_measurements(self, vertices, faces, keypoints_3d, height_cm):
        import trimesh
        import numpy as np

        slices_3d = {}
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_height = vertices[:, 1].max() - vertices[:, 1].min()
        scale = height_cm / mesh_height

        def get_kp(name):
            return keypoints_3d[KEYPOINTS[name]]

        def dist(kp1, kp2):
            return float(np.linalg.norm(get_kp(kp1) - get_kp(kp2)))

        def extract_slice(y_level):
            try:
                return mesh.section(plane_origin=[0, y_level, 0], plane_normal=[0, 1, 0])
            except:
                return None

        def get_torso_polygon(slice_3d, name=""):
            try:
                if slice_3d is None:
                    return None, None
                slice_2d, to_3d = slice_3d.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None, None
                best_poly = min(slice_2d.polygons_full,
                               key=lambda p: p.centroid.x**2 + p.centroid.y**2)
                coords_2d = np.array(best_poly.exterior.coords)
                coords_3d_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), np.ones(len(coords_2d))])
                coords_3d = (to_3d @ coords_3d_h.T).T[:, :3]
                return best_poly.exterior.length, coords_3d.tolist()
            except:
                return None, None

        def get_leg_circ(y_level, name=""):
            try:
                s = extract_slice(y_level)
                if s is None:
                    return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None
                if len(slice_2d.polygons_full) == 1:
                    poly = slice_2d.polygons_full[0]
                    coords_2d = np.array(poly.exterior.coords)
                    coords_3d_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), np.ones(len(coords_2d))])
                    coords_3d = (to_3d @ coords_3d_h.T).T[:, :3]
                    slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                    return poly.exterior.length / 2
                sorted_polys = sorted(slice_2d.polygons_full, key=lambda p: p.area, reverse=True)
                chosen = None
                for p in sorted_polys:
                    if abs(p.centroid.x) > 0.02:
                        chosen = p
                        break
                if chosen is None:
                    chosen = sorted_polys[1] if len(sorted_polys) >= 2 else sorted_polys[0]
                coords_2d = np.array(chosen.exterior.coords)
                coords_3d_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), np.ones(len(coords_2d))])
                coords_3d = (to_3d @ coords_3d_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                return chosen.exterior.length
            except:
                return None

        def get_limb_circ(y_level, name=""):
            try:
                s = extract_slice(y_level)
                if s is None:
                    return None
                slice_2d, to_3d = s.to_planar()
                if not hasattr(slice_2d, 'polygons_full') or not slice_2d.polygons_full:
                    return None
                if len(slice_2d.polygons_full) == 1:
                    poly = slice_2d.polygons_full[0]
                    coords_2d = np.array(poly.exterior.coords)
                    coords_3d_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), np.ones(len(coords_2d))])
                    coords_3d = (to_3d @ coords_3d_h.T).T[:, :3]
                    slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                    return poly.exterior.length / 2
                best = max(slice_2d.polygons_full, key=lambda p: p.centroid.x**2 + p.centroid.y**2)
                coords_2d = np.array(best.exterior.coords)
                coords_3d_h = np.column_stack([coords_2d, np.zeros(len(coords_2d)), np.ones(len(coords_2d))])
                coords_3d = (to_3d @ coords_3d_h.T).T[:, :3]
                slices_3d[name] = {"y": float(y_level), "contour": coords_3d.tolist()}
                return best.exterior.length
            except:
                return None

        measurements = {}
        shoulder_y = (get_kp('left_shoulder')[1] + get_kp('right_shoulder')[1]) / 2
        hip_y = (get_kp('left_hip')[1] + get_kp('right_hip')[1]) / 2
        knee_y = (get_kp('left_knee')[1] + get_kp('right_knee')[1]) / 2
        ankle_y = (get_kp('left_ankle')[1] + get_kp('right_ankle')[1]) / 2
        elbow_y = (get_kp('left_elbow')[1] + get_kp('right_elbow')[1]) / 2

        # Torso circumferences
        for name, frac in [("chest", 0.25), ("underbust", 0.40), ("waist", 0.55)]:
            y = shoulder_y - (shoulder_y - hip_y) * frac
            s = extract_slice(y)
            c, contour = get_torso_polygon(s, name)
            if c:
                measurements[name] = round(c * scale, 1)
                if contour:
                    slices_3d[name] = {"y": float(y), "contour": contour}

        s = extract_slice(hip_y)
        c, contour = get_torso_polygon(s, "hips")
        if c:
            measurements["hips"] = round(c * scale, 1)
            if contour:
                slices_3d["hips"] = {"y": float(hip_y), "contour": contour}

        # Legs
        y_thigh = hip_y - (hip_y - knee_y) * 0.35
        c = get_leg_circ(y_thigh, "thigh")
        if c:
            measurements["thigh"] = round(c * scale, 1)

        c = get_leg_circ(knee_y, "knee")
        if c:
            measurements["knee"] = round(c * scale, 1)

        y_calf = knee_y - (knee_y - ankle_y) * 0.30
        c = get_leg_circ(y_calf, "calf")
        if c:
            measurements["calf"] = round(c * scale, 1)

        # Arms
        y_bic = shoulder_y - (shoulder_y - elbow_y) * 0.40
        c = get_limb_circ(y_bic, "biceps")
        if c:
            measurements["biceps"] = round(c * scale, 1)

        # Widths
        shoulder_slice_y = shoulder_y - (shoulder_y - hip_y) * 0.05
        try:
            s = extract_slice(shoulder_slice_y)
            if s is not None:
                verts = np.array(s.vertices)
                sw = (verts[:, 0].max() - verts[:, 0].min()) * scale
                measurements["shoulder_width"] = round(sw, 1)
        except:
            measurements["shoulder_width"] = round(dist('left_shoulder', 'right_shoulder') * scale, 1)

        # Lengths
        left_arm = dist('left_shoulder', 'left_elbow') + dist('left_elbow', 'left_wrist')
        right_arm = dist('right_shoulder', 'right_elbow') + dist('right_elbow', 'right_wrist')
        measurements["arm_length"] = round(((left_arm + right_arm) / 2) * scale, 1)
        measurements["inseam"] = round(abs(hip_y - ankle_y) * scale, 1)
        measurements["torso_length"] = round(abs(shoulder_y - hip_y) * scale, 1)

        return measurements, slices_3d


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
    return {"status": "ok", "service": "ETEKA Body Scan WIX API"}


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
