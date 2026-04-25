import { NextRequest, NextResponse } from "next/server";

const MODAL_API_URL = process.env.NEXT_PUBLIC_MODAL_API_URL || process.env.MODAL_API_URL || "";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const mode = formData.get("mode") as string;
    const heightCm = parseFloat((formData.get("height_cm") as string) || "170");

    if (!MODAL_API_URL) {
      return NextResponse.json(
        { error: "Backend non configure (MODAL_API_URL manquant)" },
        { status: 500 }
      );
    }

    // Build Modal FormData based on mode
    const modalFormData = new FormData();
    modalFormData.append("height_cm", heightCm.toString());

    if (mode === "standard") {
      const poses = ["front", "left", "back", "right"];
      for (const pose of poses) {
        const file = formData.get(`photo_${pose}`) as File | null;
        if (!file) {
          return NextResponse.json(
            { error: `Photo ${pose} manquante` },
            { status: 400 }
          );
        }
        if (file.size > 15 * 1024 * 1024) {
          return NextResponse.json(
            { error: `Photo ${pose} trop volumineuse (max 15 Mo)` },
            { status: 400 }
          );
        }
        modalFormData.append(`photo_${pose}`, file);
      }
      const endpoint = `${MODAL_API_URL}/analyze_multiview`;
      const response = await fetch(endpoint, { method: "POST", body: modalFormData });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return NextResponse.json(
          { error: errorData.detail || "Erreur backend" },
          { status: response.status }
        );
      }

      const result = await response.json();
      delete result.mesh_glb_base64;
      delete result.debug;
      return NextResponse.json(result);
    }

    if (mode === "premium") {
      const video = formData.get("video") as File | null;
      if (!video) {
        return NextResponse.json({ error: "Video manquante" }, { status: 400 });
      }
      if (video.size > 100 * 1024 * 1024) {
        return NextResponse.json(
          { error: "Video trop volumineuse (max 100 Mo)" },
          { status: 400 }
        );
      }
      modalFormData.append("video", video);
      const endpoint = `${MODAL_API_URL}/analyze_video`;
      const response = await fetch(endpoint, { method: "POST", body: modalFormData });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return NextResponse.json(
          { error: errorData.detail || "Erreur backend" },
          { status: response.status }
        );
      }

      const result = await response.json();
      delete result.mesh_glb_base64;
      delete result.debug;
      return NextResponse.json(result);
    }

    return NextResponse.json({ error: "Mode invalide" }, { status: 400 });
  } catch (err) {
    console.error("Analyze error:", err);
    return NextResponse.json(
      { error: "Erreur interne du serveur" },
      { status: 500 }
    );
  }
}
