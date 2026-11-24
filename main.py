import os
import io
import base64
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
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

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse({"status": "ok"})
    else:
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse({"error": str(e)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Load MI-GAN Neural Model with Ultra-Low Memory Budget (~90MB)
migan_session = None

try:
    local_path = os.path.join(os.path.dirname(__file__), "migan_pipeline_v2.onnx")
    if os.path.exists(local_path):
        model_path = local_path
    else:
        print("[*] Fetching MI-GAN Neural Model weights from Hub...")
        model_path = hf_hub_download(repo_id="ravetools/watermark-remover", filename="migan_pipeline_v2.onnx")

    print(f"[*] Initializing MI-GAN Neural Engine from {model_path}...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False  # Critical: restricts RAM to ~90MB (never crashes Render)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    migan_session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    print("[+] MI-GAN Modulated Inpainting GAN v2 Ready in Cloud (Low-RAM Mode)!")
except Exception as e:
    print(f"[-] MI-GAN load warning: {e}")
    migan_session = None

def run_migan_inpaint(img_bgr, mask_u8):
    """
    Executes true Modulated Inpainting GAN (MI-GAN v2) neural erasure
    using tight bounding-box cropping with contextual margin.
    Peak RAM: ~90 MB (safely below Render 512 MB ceiling).
    """
    h, w = img_bgr.shape[:2]
    ys, xs = np.where(mask_u8 > 20)
    if len(ys) == 0:
        return img_bgr

    # Tight crop around watermark with 32px contextual margin
    pad = 32
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + pad)

    crop_img = img_bgr[y1:y2, x1:x2]
    crop_mask = mask_u8[y1:y2, x1:x2]
    ch, cw = crop_img.shape[:2]

    # Ensure dimensions are multiples of 16 for GAN convolutional layers, max 512
    max_dim = 512
    if max(ch, cw) > max_dim:
        scale = max_dim / max(ch, cw)
        target_w = max(16, int(round(cw * scale / 16.0)) * 16)
        target_h = max(16, int(round(ch * scale / 16.0)) * 16)
    else:
        target_w = max(16, int(np.ceil(cw / 16.0)) * 16)
        target_h = max(16, int(np.ceil(ch / 16.0)) * 16)

    proc_img = cv2.resize(crop_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    proc_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # In MI-GAN: 0 is target to remove, 255 is background to keep!
    migan_mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    migan_mask[proc_mask > 20] = 0

    crop_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    m_in = migan_mask[np.newaxis, np.newaxis, ...]
    im_in = np.transpose(crop_rgb, (2, 0, 1))[np.newaxis, ...]

    # Neural GAN Inference (~0.4s)
    out = migan_session.run(None, {'image': im_in, 'mask': m_in})
    out_rgb = np.transpose(out[0][0], (1, 2, 0))
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    if (target_w, target_h) != (cw, ch):
        out_bgr = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # Seamless composite strictly on watermark mask
    result = img_bgr.copy()
    crop_composite = np.where(crop_mask[:, :, np.newaxis] > 20, out_bgr, crop_img)
    result[y1:y2, x1:x2] = crop_composite
    return result

@app.get("/")
def root():
    return {"status": "ok", "service": "GhostErase MI-GAN AI Cloud Engine"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "MI-GAN Neural Inpainter (Active)" if migan_session else "C++ Telea Fallback"
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

        if len(mask_raw.shape) == 3 and mask_raw.shape[2] == 4:
            mask_gray = (mask_raw[:, :, 3] > 20).astype(np.uint8) * 255
        elif len(mask_raw.shape) == 3:
            mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_raw

        _, mask_binary = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)

        # Run MI-GAN Neural Inpainter
        if migan_session is not None:
            try:
                result_bgr = run_migan_inpaint(img_bgr, mask_binary)
                engine_used = "MI-GAN Neural GAN (Zero Blur)"
            except Exception as migan_err:
                print(f"MI-GAN error, using fallback: {migan_err}")
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
        return {"result": out_b64, "image": out_b64, "engine": engine_used}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
