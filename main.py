import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import torch
from torchsr.models import carn_m  # lightweight SR model
import easyocr

# --------------- FastAPI setup ---------------
app = FastAPI()

# --------------- Load SR model ---------------
device = torch.device("cpu")
sr_model = carn_m(scale=2, pretrained=True).to(device)
sr_model.eval()

# --------------- OCR Reader ---------------
ocr_reader = easyocr.Reader(['en'], gpu=False)

# --------------- Helper Functions ---------------
def enhance_image(pil_img: Image.Image) -> Image.Image:
    """Enhance a PIL image using a lightweight SR model."""
    # Convert PIL to numpy, normalize
    img_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    # Convert to tensor (C, H, W)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = sr_model(img_t)
    # Convert back to numpy in uint8
    out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)

def to_base64(pil_img: Image.Image) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def read_plate_text(pil_img: Image.Image) -> str:
    """Extract text from plate using EasyOCR."""
    arr = np.array(pil_img)
    result = ocr_reader.readtext(arr)
    texts = [item[1] for item in result]
    return " ".join(texts) if texts else ""

# --------------- Endpoints ---------------
@app.get("/")
def home():
    return {"status": 200, "message": "API is running"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": 400, "message": "Invalid image", "data": {}}

        # Get PIL version of full image
        original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Number plate detection (OpenCV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        plate_area = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                plate_area = approx
                break

        if plate_area is None:
            return {"status": 400, "message": "Number plate not detected", "data": {}}

        x, y, w, h = cv2.boundingRect(plate_area)
        plate_crop = img[y:y+h, x:x+w]
        plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

        # Enhance both images
        enhanced_plate = enhance_image(plate_pil)
        enhanced_vehicle = enhance_image(original_pil)

        # OCR to read plate text
        plate_text = read_plate_text(plate_pil)

        return {
            "status": 200,
            "message": "Image enhanced and text recognized",
            "data": {
                "number_plate": to_base64(enhanced_plate),
                "vehicle": to_base64(enhanced_vehicle),
                "plate_text": plate_text
            }
        }

    except Exception as e:
        return {"status": 500, "message": f"Internal server error: {str(e)}", "data": {}}
