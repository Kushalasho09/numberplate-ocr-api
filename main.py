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
# Load once to avoid repeated downloads
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

# ---------------------------
# Helper Functions
# ---------------------------
def enhance_image(pil_img: Image.Image) -> Image.Image:
    """
    Enhance a PIL image using the EdsrModel from super_image.
    Ensures proper dtype and channel format.
    """
    # Ensure PIL image
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(pil_img)

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Ensure dtype uint8
    img_np = np.array(pil_img).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # Load into super_image tensor
    tensor_img = ImageLoader.load_image(pil_img)

    # Run model
    preds = model(tensor_img)

    # Convert back to PIL Image
    enhanced_pil = ImageLoader.save_image(preds)
    return enhanced_pil

def to_base64(pil_img: Image.Image) -> str:
    """Convert PIL Image to Base64 string"""
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

# ---------------------------
# Root Check Endpoint
# ---------------------------
@app.get("/")
def home():
    return {"status": 200, "message": "OCR + Enhancement API running"}

# ---------------------------
# Main Processing Route
# ---------------------------
@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": 400, "message": "Invalid image", "data": {}}

        # PIL version of the full image
        original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ------------------------------
        # Number Plate Detection
        # ------------------------------
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
            return {"status": 404, "message": "Number plate not detected", "data": {}}

        # Crop number plate
        x, y, w, h = cv2.boundingRect(plate_area)
        plate_crop = img[y:y+h, x:x+w]
        plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

        # ------------------------------
        # Enhance Images
        # ------------------------------
        enhanced_plate = enhance_image(plate_pil)
        enhanced_vehicle = enhance_image(original_pil)

        # ------------------------------
        # Return Base64 Images
        # ------------------------------
        return {
            "status": 200,
            "message": "Image enhanced successfully",
            "data": {
                "number_plate": to_base64(enhanced_plate),
                "vehicle": to_base64(enhanced_vehicle)
            }
        }

    except Exception as e:
        return {"status": 500, "message": f"Internal server error: {str(e)}", "data": {}}
