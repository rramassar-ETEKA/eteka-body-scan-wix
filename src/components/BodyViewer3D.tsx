"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

interface SliceData {
  y: number;
  contour: number[][];
}

interface BodyViewer3DProps {
  vertices: number[][];
  faces: number[][];
  measurements?: Record<string, number>;
  keypoints?: Record<string, number[]>;
  slices?: Record<string, SliceData>;
}

const MEASUREMENT_LEVELS: Record<string, { yFraction: number; color: number }> = {
  Epaules: { yFraction: 0.18, color: 0xf59e0b },
  Poitrine: { yFraction: 0.25, color: 0x6366f1 },
  Taille: { yFraction: 0.38, color: 0x06b6d4 },
  Hanches: { yFraction: 0.48, color: 0x10b981 },
  Entrejambe: { yFraction: 0.75, color: 0xec4899 },
};

const KEYPOINT_LABELS: Record<string, string> = {
  nose: "Nez",
  neck: "Cou",
  left_shoulder: "Epaule G",
  right_shoulder: "Epaule D",
  left_elbow: "Coude G",
  right_elbow: "Coude D",
  left_wrist: "Poignet G",
  right_wrist: "Poignet D",
  left_hip: "Hanche G",
  right_hip: "Hanche D",
  left_knee: "Genou G",
  right_knee: "Genou D",
  left_ankle: "Cheville G",
  right_ankle: "Cheville D",
  left_acromion: "Acromion G",
  right_acromion: "Acromion D",
};

function createTextSprite(text: string, color: number, fontSize = 24): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;

  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.fillStyle = "#" + color.toString(16).padStart(6, "0");
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.4, 0.1, 1);
  return sprite;
}

const SLICE_COLORS: Record<string, number> = {
  chest: 0x6366f1,
  underbust: 0x8b5cf6,
  waist: 0x06b6d4,
  hips: 0x10b981,
  thigh: 0xf59e0b,
  knee: 0xef4444,
  calf: 0xec4899,
  biceps: 0xf97316,
};

export default function BodyViewer3D({ vertices, faces, measurements, keypoints, slices }: BodyViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const keypointGroupRef = useRef<THREE.Group | null>(null);
  const [showKeypoints, setShowKeypoints] = useState(false);

  // Toggle keypoint visibility
  useEffect(() => {
    if (keypointGroupRef.current) {
      keypointGroupRef.current.visible = showKeypoints;
    }
  }, [showKeypoints]);

  useEffect(() => {
    if (!containerRef.current || !vertices || !faces) return;
    if (vertices.length === 0 || faces.length === 0) return;

    if (cleanupRef.current) cleanupRef.current();

    try {

    const container = containerRef.current;
    const width = container.clientWidth || 500;
    const height = 480;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e293b);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
    camera.position.set(0, 0.1, 2);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(5, 10, 7);
    scene.add(directional);

    // Build mesh from vertices/faces
    const geometry = new THREE.BufferGeometry();
    const vertArray = new Float32Array(vertices.flat());
    geometry.setAttribute("position", new THREE.BufferAttribute(vertArray, 3));

    const indexArray = new Uint32Array(faces.flat());
    geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
    geometry.computeVertexNormals();

    // Compute center offset before centering (needed for keypoints)
    const tempBox = new THREE.Box3().setFromBufferAttribute(
      geometry.getAttribute("position") as THREE.BufferAttribute
    );
    const meshCenter = tempBox.getCenter(new THREE.Vector3());

    geometry.center();

    const material = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      metalness: 0.1,
      roughness: 0.6,
      side: THREE.DoubleSide,
    });

    const bodyGroup = new THREE.Group();
    const mesh = new THREE.Mesh(geometry, material);
    bodyGroup.add(mesh);

    // Keypoints
    const keypointGroup = new THREE.Group();
    keypointGroupRef.current = keypointGroup;
    keypointGroup.visible = showKeypoints;

    if (keypoints) {
      const sphereGeom = new THREE.SphereGeometry(0.008, 8, 8);

      for (const [name, pos] of Object.entries(keypoints)) {
        if (!pos || pos.length < 3) continue;

        // Offset keypoint positions to match centered geometry
        const kpMat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
        const sphere = new THREE.Mesh(sphereGeom, kpMat);
        sphere.position.set(
          pos[0] - meshCenter.x,
          pos[1] - meshCenter.y,
          pos[2] - meshCenter.z
        );
        keypointGroup.add(sphere);

        // Label
        const label = KEYPOINT_LABELS[name] || name;
        const sprite = createTextSprite(label, 0xff6666, 18);
        sprite.position.set(
          pos[0] - meshCenter.x + 0.05,
          pos[1] - meshCenter.y + 0.015,
          pos[2] - meshCenter.z
        );
        sprite.scale.set(0.25, 0.06, 1);
        keypointGroup.add(sprite);
      }
    }

    bodyGroup.add(keypointGroup);

    // Center and flip
    const box = new THREE.Box3().setFromObject(mesh);
    const center = box.getCenter(new THREE.Vector3());
    bodyGroup.position.sub(center);
    bodyGroup.scale.y = -1;

    const pivotGroup = new THREE.Group();
    pivotGroup.add(bodyGroup);
    pivotGroup.rotation.y = Math.PI;
    scene.add(pivotGroup);

    // Measurement lines
    const size = box.getSize(new THREE.Vector3());

    for (const [label, config] of Object.entries(MEASUREMENT_LEVELS)) {
      const y = size.y / 2 - size.y * config.yFraction;

      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-size.x * 0.6, y, 0),
        new THREE.Vector3(size.x * 0.6, y, 0),
      ]);
      const lineMat = new THREE.LineBasicMaterial({ color: config.color, transparent: true, opacity: 0.5 });
      const line = new THREE.Line(lineGeom, lineMat);
      pivotGroup.add(line);

      let text = label;
      if (measurements) {
        const keyMap: Record<string, string> = {
          Epaules: "shoulder_width",
          Poitrine: "chest",
          Taille: "waist",
          Hanches: "hips",
          Entrejambe: "inseam",
        };
        const key = keyMap[label];
        if (key && measurements[key]) {
          text = `${label}: ${measurements[key]} cm`;
        }
      }

      const sprite = createTextSprite(text, config.color);
      sprite.position.set(size.x * 0.7, y, 0);
      pivotGroup.add(sprite);
    }

    // Slice contours (circumference visualization)
    if (slices) {
      for (const [name, sliceData] of Object.entries(slices)) {
        try {
          if (!sliceData.contour || sliceData.contour.length < 3) continue;

          const color = SLICE_COLORS[name] || 0xffffff;

          const points = sliceData.contour.map(
            (p) => new THREE.Vector3(p[0] - meshCenter.x, p[1] - meshCenter.y, p[2] - meshCenter.z)
          );

          const curve = new THREE.CatmullRomCurve3(points, true);
          const tubeGeom = new THREE.TubeGeometry(curve, Math.max(points.length * 2, 32), 0.003, 6, true);
          const tubeMat = new THREE.MeshBasicMaterial({ color });
          const tube = new THREE.Mesh(tubeGeom, tubeMat);
          bodyGroup.add(tube);

          const shape = new THREE.Shape();
          const yLevel = sliceData.contour[0][1] - meshCenter.y;
          shape.moveTo(sliceData.contour[0][0] - meshCenter.x, sliceData.contour[0][2] - meshCenter.z);
          for (let i = 1; i < sliceData.contour.length; i++) {
            shape.lineTo(sliceData.contour[i][0] - meshCenter.x, sliceData.contour[i][2] - meshCenter.z);
          }
          shape.closePath();

          const shapeGeom = new THREE.ShapeGeometry(shape);
          const shapeMat = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide,
          });
          const shapeMesh = new THREE.Mesh(shapeGeom, shapeMat);
          shapeMesh.rotation.x = -Math.PI / 2;
          shapeMesh.position.y = yLevel;
          bodyGroup.add(shapeMesh);
        } catch (err) {
          console.warn(`[Body Scan] Slice ${name} render error:`, err);
        }
      }
    }

    // Mouse rotation
    let isDragging = false;
    let prevX = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevX = e.clientX;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const delta = e.clientX - prevX;
      pivotGroup.rotation.y += delta * 0.01;
      prevX = e.clientX;
    };
    const onMouseUp = () => {
      isDragging = false;
    };

    const onTouchStart = (e: TouchEvent) => {
      isDragging = true;
      prevX = e.touches[0].clientX;
    };
    const onTouchMove = (e: TouchEvent) => {
      if (!isDragging) return;
      const delta = e.touches[0].clientX - prevX;
      pivotGroup.rotation.y += delta * 0.01;
      prevX = e.touches[0].clientX;
    };
    const onTouchEnd = () => {
      isDragging = false;
    };

    renderer.domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("touchstart", onTouchStart);
    window.addEventListener("touchmove", onTouchMove);
    window.addEventListener("touchend", onTouchEnd);

    // Animation loop
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    cleanupRef.current = () => {
      cancelAnimationFrame(animationId);
      renderer.domElement.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      renderer.domElement.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
    };

    return () => {
      if (cleanupRef.current) cleanupRef.current();
    };
    } catch (err) {
      console.error("[Body Scan] Viewer3D init error:", err);
    }
  }, [vertices, faces, measurements, keypoints, slices, showKeypoints]);

  return (
    <div className="glass rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between px-4 pt-3">
        <span className="text-sm text-[var(--foreground)]/60">Modele 3D</span>
        <button
          onClick={() => setShowKeypoints(!showKeypoints)}
          className={`text-xs px-3 py-1 rounded-lg transition-all ${
            showKeypoints
              ? "bg-red-500/20 text-red-400 border border-red-500/30"
              : "bg-[var(--surface)] text-[var(--foreground)]/50 border border-[var(--border)]"
          }`}
        >
          {showKeypoints ? "Masquer keypoints" : "Afficher keypoints"}
        </button>
      </div>
      <div
        ref={containerRef}
        className="w-full cursor-grab active:cursor-grabbing"
        style={{ minHeight: 480 }}
      />
      <p className="text-center text-xs text-[var(--foreground)]/40 py-2">
        Glissez pour tourner le modele 3D
      </p>
    </div>
  );
}
