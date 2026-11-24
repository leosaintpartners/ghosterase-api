import os
import io
import base64
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="GhostErase AI - MI-GAN Neural Inpainting Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MI-GAN Neural Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "migan_pipeline_v2.onnx")
migan_session = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"[*] Loading MI-GAN Neural Engine from {MODEL_PATH}...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        migan_session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
        print("[+] MI-GAN Neural Model Ready!")
    else:
        print(f"[-] MI-GAN model not found at {MODEL_PATH}")
except Exception as e:
    print(f"[-] Error loading MI-GAN: {e}")
    migan_session = None

def run_migan_inpaint(img_bgr, mask_u8):
    """
    Executes true Modulated Inpainting GAN (MI-GAN v2) neural erasure.
    img_bgr: (H, W, 3) uint8 image in BGR
    mask_u8: (H, W) uint8 mask (255 = watermark area, 0 = background)
    """
    h, w = img_bgr.shape[:2]
    user_mask_bool = (mask_u8 > 20)

    # In MI-GAN: 0 is target to remove, 255 is background to keep!
    migan_mask = np.full((h, w), 255, dtype=np.uint8)
    migan_mask[user_mask_bool] = 0

    crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    migan_mask_in = migan_mask[np.newaxis, np.newaxis, ...]
    img_in = np.transpose(crop_rgb, (2, 0, 1))[np.newaxis, ...]

    # Neural inference
    out = migan_session.run(None, {'image': img_in, 'mask': migan_mask_in})
    out_rgb = np.transpose(out[0][0], (1, 2, 0))
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    # Forensic grain & texture blending
    border_ring = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) - mask_u8
    if np.count_nonzero(border_ring) > 40:
        high_pass = img_bgr.astype(np.float32) - cv2.GaussianBlur(img_bgr, (5, 5), 0).astype(np.float32)
        noise_sigma = np.std(high_pass[border_ring > 0], axis=0)
        grain = np.random.normal(0, np.clip(noise_sigma, 0.6, 6.0), img_bgr.shape)
        out_bgr_textured = np.clip(out_bgr.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    else:
        out_bgr_textured = out_bgr

    # Seamless composite
    final_bgr = np.where(mask_u8[:, :, np.newaxis] > 20, out_bgr_textured, img_bgr)
    return final_bgr

@app.get("/")
def root():
    return {"status": "ok", "service": "GhostErase MI-GAN AI Engine"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "MI-GAN Modulated Inpainting GAN v2" if migan_session else "OpenCV Telea Fallback"
    }

@app.post("/api/inpaint")
async def inpaint_endpoint(request: Request):
    try:
        data = await request.json()
        img_b64 = data.get("image", "")
        mask_b64 = data.get("mask", "")

        if not img_b64 or not mask_b64:
            return JSONResponse({"error": "Missing image or mask data"}, status_code=400)

        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        if "," in mask_b64:
            mask_b64 = mask_b64.split(",", 1)[1]

        # Decode Image
        img_bytes = base64.b64decode(img_b64)
        nparr_img = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"error": "Failed to decode input image"}, status_code=400)

        # Decode Mask
        mask_bytes = base64.b64decode(mask_b64)
        nparr_mask = np.frombuffer(mask_bytes, np.uint8)
        mask_raw = cv2.imdecode(nparr_mask, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            return JSONResponse({"error": "Failed to decode mask"}, status_code=400)

        # Convert mask to single channel 8-bit grayscale
        if len(mask_raw.shape) == 3 and mask_raw.shape[2] == 4:
            mask_gray = (mask_raw[:, :, 3] > 20).astype(np.uint8) * 255
        elif len(mask_raw.shape) == 3:
            mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_raw

        _, mask_binary = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)

        # Execute MI-GAN if available
        if migan_session is not None:
            try:
                result_bgr = run_migan_inpaint(img_bgr, mask_binary)
                engine_used = "MI-GAN Neural GAN"
            except Exception as migan_err:
                print(f"MI-GAN run error, falling back: {migan_err}")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
                result_bgr = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
                engine_used = "C++ Telea Fallback"
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            result_bgr = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            engine_used = "C++ Telea Engine"

        # Encode back to PNG
        success, buf = cv2.imencode(".png", result_bgr)
        if not success:
            return JSONResponse({"error": "Failed to encode output image"}, status_code=500)

        out_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
        return {"result": out_b64, "engine": engine_used}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
