import numpy as np
from PIL import Image, ImageDraw
from skimage import color, filters
from skimage.feature import graycomatrix, graycoprops
from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize FastAPI
app = FastAPI()

# Tumor classifier model (output is 'tumor' or 'non_tumor')
model = RandomForestClassifier(n_estimators=100, random_state=42)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Prediction API! Use the /predict-tumor/ endpoint to upload brain MRI images."}

@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Extract features from the image
        image_features = extract_image_features(image_array)
        
        # Reshape the features for prediction
        features = np.array(image_features).reshape(1, -1)
        
        # Predict the tumor type ('tumor' or 'non_tumor')
        prediction = model.predict(features)
        predicted_label = prediction[0]
        
        # Highlight the tumor region if tumor is detected
        if predicted_label == 'tumor':
            image_with_tumor = highlight_tumor(image_array)
            plots = generate_plots(image_array, image_with_tumor, image_features)
        else:
            plots = generate_plots(image_array, None, image_features)
        
        return {
            "predicted_label": predicted_label,
            "plots": plots
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while processing the image: {e}")

# Training the model with dummy data for demonstration
def train_model():
    # Generate dummy data for training
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.choice(['tumor', 'non_tumor'], size=100)  # Random labels 'tumor' or 'non_tumor'

    # Split the dummy data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

    # Fit the model on dummy data
    model.fit(X_train, y_train)

# Call the training function when the application starts
train_model()

def extract_image_features(image: np.array):
    """Extract features from a given image array."""
    if image.shape[-1] == 4:  # Handle RGBA images
        image = color.rgba2rgb(image)

    # Convert to grayscale and normalize
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))

    # Extract intensity features
    mean_intensity = np.mean(image_normalized)
    std_intensity = np.std(image_normalized)
    cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # Extract edge strength using Sobel filter
    edges_sobel = filters.sobel(image_normalized)
    avg_edge_strength = np.mean(edges_sobel)
    
    # Extract GLCM texture features
    glcm = graycomatrix((image_normalized * 255).astype(np.uint8), distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    return [mean_intensity, std_intensity, cv, avg_edge_strength, contrast, 
            dissimilarity, homogeneity, energy, correlation, asm]

def highlight_tumor(image: np.array):
    """Highlight tumor regions on the image by detecting areas with higher contrast and edge strength."""
    image_gray = color.rgb2gray(image)
    
    # Detect edges using Sobel filter
    edges = filters.sobel(image_gray)
    
    # Thresholding to create a mask for the tumor
    tumor_mask = edges > np.percentile(edges, 95)
    
    # Create an RGB image with the tumor region highlighted
    image_with_tumor = image.copy()
    image_with_tumor[tumor_mask] = [255, 0, 0]  # Highlight tumor in red
    
    return image_with_tumor

def generate_plots(original_image, tumor_image, image_features):
    plots = {}
    
    # Original image plot
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    plots['original'] = plot_to_base64(plt)
    
    # Tumor-highlighted image plot (if tumor is detected)
    if tumor_image is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(tumor_image)
        plt.title("Tumor Region Highlighted")
        plt.axis('off')
        plots['tumor_highlighted'] = plot_to_base64(plt)
    
    # Intensity Histogram plot
    image_gray = color.rgb2gray(original_image)
    plt.figure(figsize=(10, 5))
    plt.hist(image_gray.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Intensity Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid()
    plots['histogram'] = plot_to_base64(plt)
    
    # GLCM texture features plot
    glcm = graycomatrix((image_gray * 255).astype(np.uint8), distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    features = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    values = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=features, y=values)
    plt.title('GLCM Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid()
    plots['glcm_features'] = plot_to_base64(plt)
    
    return plots

def plot_to_base64(plt):
    """Convert plot to base64-encoded image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
