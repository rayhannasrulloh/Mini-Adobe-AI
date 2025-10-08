import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def get_image_path(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def save_image(img, original_filename):
    """Saves a CV2 image and returns its new filename."""
    filename = f"{uuid.uuid4().hex}_{secure_filename(original_filename)}"
    filepath = get_image_path(filename)
    # Convert color space from BGR (cv2) to RGB for saving with PIL
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(filepath)
    return filename

# --- Main Route ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = get_image_path(filename)
        file.save(filepath)
        return jsonify({'filename': filename})

# --- API Endpoints for Image Editing ---
@app.route('/api/process', methods=['POST'])
def process_image():
    data = request.get_json()
    op = data.get('operation')
    filename = data.get('filename')
    value = data.get('value')

    filepath = get_image_path(filename)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404
        
    output_img = None
    
    # --- Basic Image Editing Tools ---
    if op == 'brightness':
        output_img = cv2.convertScaleAbs(img, alpha=1, beta=int(value))
    elif op == 'contrast':
        output_img = cv2.convertScaleAbs(img, alpha=float(value), beta=0)
    elif op == 'saturation':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, int(value))
        final_hsv = cv2.merge((h, s, v))
        output_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    elif op == 'rotate':
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, int(value), 1.0)
        output_img = cv2.warpAffine(img, M, (w, h))
    elif op == 'flip':
        flip_code = 1 if value == 'horizontal' else 0
        output_img = cv2.flip(img, flip_code)
    elif op == 'histogram-equalization':
        if len(img.shape) > 2: # Color image
             ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
             ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
             output_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else: # Grayscale image
            output_img = cv2.equalizeHist(img)
    elif op == 'gaussian-blur':
        ksize = int(value)
        if ksize % 2 == 0: ksize += 1 # Kernel size must be odd
        output_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif op == 'median-blur':
        ksize = int(value)
        if ksize % 2 == 0: ksize += 1 # Kernel size must be odd
        output_img = cv2.medianBlur(img, ksize)
    elif op == 'sharpen':
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        output_img = cv2.filter2D(img, -1, kernel)
    elif op == 'edge-detection': # Sobel Edge Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        output_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif op == 'erosion':
        kernel = np.ones((5, 5), np.uint8)
        output_img = cv2.erode(img, kernel, iterations=1)
    elif op == 'dilation':
        kernel = np.ones((5, 5), np.uint8)
        output_img = cv2.dilate(img, kernel, iterations=1)
    elif op == 'threshold':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, output_img = cv2.threshold(gray, int(value), 255, cv2.THRESH_BINARY)
        
    # --- AI-Embedded Features ---
    elif op == 'remove-bg':
        # rembg expects a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        output_pil = remove(pil_img)
        # Convert back to CV2 format
        output_img = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGBA_BGRA)
    elif op == 'content-aware-fill':
        # This is a placeholder for a more complex implementation
        # where the user would provide a mask.
        # For this example, we'll just return the original image.
        output_img = img # Replace with actual inpainting logic
    elif op == 'style-transfer':
        # Requires pre-trained model loading, which is complex.
        # Placeholder for demonstration.
        output_img = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
    elif op == 'super-resolution':
        # Requires a DNN model. Placeholder.
        # For demonstration, we'll just resize using Lanczos interpolation
        output_img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    elif op == 'beautify':
        # Simple skin smoothing example
        output_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # --- Save and Return Result ---
    if output_img is not None:
        new_filename = save_image(output_img, filename)
        return jsonify({'filename': new_filename})
    else:
        return jsonify({'error': 'Invalid operation'}), 400

if __name__ == '__main__':
    app.run(debug=True)