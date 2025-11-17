import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from super_image import EdsrModel, ImageLoader

# ---------------------------
# Create FastAPI App
# ---------------------------
app = FastAPI()

# ---------------------------
# Load Super Resolution Model
# ---------------------------
# IMPORTANT: Load only once (fast & Render-safe)
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

# ---------------------------
# Helper Functions
# ---------------------------
def enhance_image(pil_img):
    """Enhance image using ESRGAN (CPU)"""
    img = ImageLoader.load_image(pil_img)
    preds = model(img)
    out = ImageLoader.save_image(preds)
    return out

def to_base64(pil_img):
    """Convert PIL Image to Base64"""
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------
# Root Check Endpoint
# ---------------------------
@app.get("/")
def home():
    return {"status": "running", "message": "OCR + Enhancement API working"}


# ---------------------------
# Main Processing Route
# ---------------------------
@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Create PIL for full-image enhancement
    original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------
    #          NUMBER PLATE DETECTION
    # ----------------------------------------------
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
        return {"error": "Number plate not detected"}

    # Crop number plate
    x, y, w, h = cv2.boundingRect(plate_area)
    plate_crop = img[y:y+h, x:x+w]
    plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------
    #          IMAGE ENHANCEMENT
    # ----------------------------------------------
    enhanced_plate = enhance_image(plate_pil)          # cropped plate
    enhanced_vehicle = enhance_image(original_pil)     # whole image

    # ----------------------------------------------
    #          RETURN 2 IMAGES (BASE64)
    # ----------------------------------------------
    return {
        "number_plate": to_base64(enhanced_plate),
        "vehicle": to_base64(enhanced_vehicle),
    }


# ---------------------------
# Render Deployment Note
# (DO NOT CHANGE ANYTHING HERE)
# ---------------------------
# Uvicorn will be launched by Render using:
# uvicorn main:app --host 0.0.0.0 --port $PORT
