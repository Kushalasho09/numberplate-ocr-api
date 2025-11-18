import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import gc

# --------------- FastAPI setup ---------------
app = FastAPI()

# --------------- Lazy-loaded models ---------------
sr_model = None
ocr_reader = None
device = None

def get_sr_model():
    global sr_model, device
    if sr_model is None:
        import torch
        from torchsr.models import carn_m
        device = torch.device("cpu")
        sr_model = carn_m(scale=2, pretrained=True).to(device)
        sr_model.eval()
    return sr_model

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    return ocr_reader

# --------------- Helper Functions ---------------
MAX_SIZE = 512  # smaller to save memory

def resize_image(pil_img: Image.Image) -> Image.Image:
    pil_img.thumbnail((MAX_SIZE, MAX_SIZE))
    return pil_img

def enhance_image(pil_img: Image.Image, model) -> Image.Image:
    """Enhance a PIL image using a lightweight SR model."""
    import torch
    img_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(img_t)
    out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)

def to_base64(pil_img: Image.Image) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def read_plate_text(pil_img: Image.Image, reader) -> str:
    arr = np.array(pil_img)
    result = reader.readtext(arr)
    texts = [item[1] for item in result]
    return " ".join(texts) if texts else ""

# --------------- Endpoints ---------------
@app.get("/")
def home():
    return {"status": 200, "message": "API is running"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": 400, "message": "Invalid image", "data": {}}

        # 1️⃣ Extract number plate
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
            del img
            gc.collect()
            return {"status": 400, "message": "Number plate not detected", "data": {}}

        x, y, w, h = cv2.boundingRect(plate_area)
        plate_crop = img[y:y+h, x:x+w]
        del img, gray, blur, edged, cnts  # free memory
        gc.collect()

        plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        plate_pil = resize_image(plate_pil)
        del plate_crop
        gc.collect()

        # 2️⃣ Enhance plate
        sr = get_sr_model()
        enhanced_plate = enhance_image(plate_pil, sr)
        del plate_pil
        gc.collect()

        # 3️⃣ OCR
        reader = get_ocr_reader()
        plate_text = read_plate_text(enhanced_plate, reader)

        # 4️⃣ Convert to base64 and free memory
        plate_base64 = to_base64(enhanced_plate)
        del enhanced_plate
        gc.collect()

        response_data = {
            "number_plate": plate_base64,
            "plate_text": plate_text
        }

        return {"status": 200, "message": "Number plate processed successfully", "data": response_data}

    except Exception as e:
        return {"status": 500, "message": f"Internal server error: {str(e)}", "data": {}}
