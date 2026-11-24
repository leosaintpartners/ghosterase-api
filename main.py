import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="GhostErase AI - Python Inpainting Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "GhostErase AI Python Engine"}

@app.get("/health")
def health():
    return {"status": "ok", "engine": "OpenCV C++ Telea Fast Marching Inpainting"}

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

        # Threshold mask
        _, mask_binary = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)

        # Dilation for complete coverage of watermark edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)

        # High-precision C++ Telea Fast Marching Inpainting (Zero Blur)
        result_bgr = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        # Encode back to PNG
        success, buf = cv2.imencode(".png", result_bgr)
        if not success:
            return JSONResponse({"error": "Failed to encode output image"}, status_code=500)

        out_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
        return {"result": out_b64, "engine": "Python C++ Telea Fast Marching Engine"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
