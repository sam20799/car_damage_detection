# 🚗 Vehicle Damage Detection App

A **deep learning-powered web application** that automatically detects and classifies vehicle damage from images. Simply drag and drop a car image, and the model will identify the type and location of damage.

![Vehicle Damage Detection App](preview.png)
*Web interface for real-time vehicle damage classification*

---

## 🎯 Overview

This application uses **Transfer Learning with ResNet50** to classify car damage into 6 categories. The model analyzes **third-quarter front or rear view** images of vehicles and identifies whether the car is normal or has specific types of damage.

### ✨ Key Features

- **Drag & Drop Interface**: Simple and intuitive web interface powered by Streamlit
- **FastAPI Backend**: Scalable REST API server for production deployments
- **Real-Time Predictions**: Instant damage classification with confidence scores
- **High Accuracy**: ~80% validation accuracy on diverse damage types
- **6 Damage Categories**: Comprehensive classification of front and rear damage
- **Multiple Model Architectures**: Tested with ResNet50, EfficientNet, and Custom CNN

---

## 📸 Important: Image Requirements

⚠️ **For best results, the input image should capture the third-quarter front or rear view of the vehicle.**

The model is specifically trained on these angles:
- **Third-quarter front view** (front-left or front-right angle)
- **Third-quarter rear view** (rear-left or rear-right angle)

Direct front/rear or side views may produce less accurate results.

---

## 🏷️ Damage Categories

The model classifies vehicles into **6 target classes**:

| Category | Description |
|----------|-------------|
| **Front Normal** | No visible damage to front section |
| **Front Crushed** | Severe crushing damage to front |
| **Front Breakage** | Parts broken or detached from front |
| **Rear Normal** | No visible damage to rear section |
| **Rear Crushed** | Severe crushing damage to rear |
| **Rear Breakage** | Parts broken or detached from rear |

---

## 🧠 Model Details

### Final Model Architecture
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Transfer Learning Strategy**: 
  - Frozen all layers except Layer 4 and final FC layer
  - Fine-tuned Layer 4 for domain-specific feature extraction
  - Custom classifier with Dropout (0.2) + Linear layer
- **Training Data**: ~1,700 images across 6 categories
- **Validation Accuracy**: **~80%**
- **Image Input Size**: 224x224 pixels
- **Framework**: PyTorch

### Model Architecture Details

```python
class carClassifierCNNResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        
        # Freeze all layers except layer4
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        # Custom classifier head
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
```

This selective unfreezing approach allows the model to:
- Retain low-level features from ImageNet pretraining
- Adapt high-level features to vehicle damage patterns
- Prevent overfitting with regularization (Dropout)

### Models Evaluated

During development, multiple architectures were tested and compared:

| Model | Description | Performance |
|-------|-------------|-------------|
| **Custom CNN** | Built from scratch with 3 convolutional blocks | Baseline performance |
| **Custom CNN + Regularization** | Added BatchNorm, Dropout, L2 regularization | Improved generalization |
| **EfficientNet** | Transfer learning with EfficientNet-B0 | Good accuracy-efficiency balance |
| **ResNet50** | Transfer learning with ResNet50 (Final choice) | **Best performance (~80%)** |

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: Random flips, rotations, color jitter
- **Regularization**: 
  - Dropout (0.2) in classifier head
  - Layer freezing (all except Layer 4 + FC)
  - Batch Normalization (inherited from ResNet50)
- **Hyperparameter Tuning**: Performed using Optuna
- **Image Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sam20799/car_damage_detection.git
   cd car_damage_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

---

## 🌐 FastAPI Server (Optional)

For production deployments or API integration, use the FastAPI backend:

### Setup FastAPI Server

1. **Navigate to the FastAPI directory**
   ```bash
   cd fastapi-server
   ```

2. **Install FastAPI dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn server:app --reload
   ```

4. **Access the API**
   - API endpoint: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`


### Streamlit App

1. **Launch the application** using `streamlit run app.py`
2. **Drag and drop** an image of a car (third-quarter view recommended)
3. **View the prediction** with damage type and confidence score
4. **Upload another image** to test different scenarios

### Programmatic Usage

```python
from model_helper import predict

# Simple prediction
damage_type = predict('path/to/car_image.jpg')
print(f"Detected Damage: {damage_type}")

# Possible outputs:
# - 'Front Normal'
# - 'Front Crushed'
# - 'Front Breakage'
# - 'Rear Normal'
# - 'Rear Crushed'
# - 'Rear Breakage'
```

The `model_helper.py` module handles:
- Lazy loading of the model (loaded only once)
- Image preprocessing (resize, normalize)
- Inference and class prediction

---

## 📊 Model Performance

### Validation Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | ~80% |
| **Training Images** | ~1,700 |
| **Classes** | 6 |
| **Model Size** | ~90 MB |

### Model Comparison

Through extensive experimentation documented in the training notebooks, ResNet50 emerged as the best performer:

- **Custom CNN**: Provided baseline understanding of the problem
- **Custom CNN + Regularization**: Improved overfitting issues
- **EfficientNet**: Offered good efficiency but slightly lower accuracy
- **ResNet50**: Best overall performance with robust feature extraction

---

## 📂 Project Structure

```
vehicle-damage-detection/
├── README.md
├── app.py                          # Streamlit web application
├── model_helper.py                 # Helper functions for model operations
├── requirements.txt                # Main project dependencies
├── temp_file.jpeg                  # Temporary storage for uploads
│
├── model/
│   └── saved_model.pth            # Trained ResNet50 model
│
├── Training/
│   ├── car_damage_detection.ipynb # Main training notebook
│   ├── HyperParameterTuning.ipynb # Hyperparameter optimization
│   └── saved_model.pth            # Training checkpoint
│
└── fastapi-server/
    ├── server.py                   # FastAPI application
    ├── model_helper.py             # Model utilities for API
    ├── requirements.txt            # API-specific dependencies
    ├── temp_file.jpeg              # API temporary storage
    └── model/
        └── saved_model.pth        # Model for API server
```

---

## 📓 Training Notebooks

### `car_damage_detection.ipynb`
Main training pipeline covering:
- Data preprocessing and augmentation
- Model architecture comparison (Custom CNN, EfficientNet, ResNet50)
- Training loop implementation
- Model evaluation and visualization
- Final model selection

### `HyperParameterTuning.ipynb`
Hyperparameter optimization using:
- Optuna for automated search
- Learning rate optimization
- Dropout rate tuning
- Batch size experiments
- Training epoch optimization

---

## 🔧 Configuration

### Model Files

The project maintains consistent model files across directories:
- `model/saved_model.pth` - Used by Streamlit app
- `fastapi-server/model/saved_model.pth` - Used by API server
- `Training/saved_model.pth` - Training checkpoint

Ensure all copies are synchronized when updating the model.

---

## 🛠️ Tech Stack

### Frontend & Interface
- **Streamlit** - Interactive web application
- **FastAPI** - REST API server
- **Uvicorn** - ASGI server

### Deep Learning
- **PyTorch** - Deep learning framework
- **TorchVision** - Computer vision utilities
- **Pretrained Models**: ResNet50, EfficientNet

### Development & Training
- **Jupyter Notebooks** - Interactive development
- **Optuna** - Hyperparameter optimization
- **NumPy, Pandas** - Data processing
- **Matplotlib, Seaborn** - Visualization

---

## 🎓 How It Works

### Streamlit App Flow
1. **Image Upload**: User uploads a car image through drag-and-drop
2. **Preprocessing**: Image is resized to 224x224 and normalized using ImageNet statistics
3. **Model Inference**: 
   - Model lazy-loaded on first prediction (stored globally)
   - ResNet50 extracts features through frozen layers
   - Layer 4 adapts features to vehicle damage domain
   - Dropout + Linear classifier predicts damage category
4. **Classification**: Returns one of 6 damage class names
5. **Results Display**: Shows predicted damage type

### Model Prediction Pipeline

```python
# From model_helper.py
def predict(img_path):
    # 1. Load and convert image to RGB
    image = Image.open(img_path).convert("RGB")
    
    # 2. Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # 3. Load model once (cached globally)
    if trained_model is None:
        model = carClassifierCNNResNet()
        model.load_state_dict(torch.load("model/saved_model.pth"))
        model.eval()
    
    # 4. Inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]
```

### FastAPI Server Flow
1. **POST Request**: Client sends image to `/predict` endpoint
2. **Temporary Storage**: Image saved to `temp_file.jpeg`
3. **Model Processing**: Calls the same `predict()` function from `model_helper.py`
4. **JSON Response**: Returns prediction as JSON

---

## 🔮 Future Enhancements

- [ ] Add support for multiple viewing angles (side, direct front/rear)
- [ ] Implement damage severity scoring (mild, moderate, severe)
- [ ] Add damage localization with bounding boxes or segmentation
- [ ] Support batch processing for multiple images
- [ ] Export damage reports as PDF
- [ ] Add cost estimation based on damage type
- [ ] Implement Grad-CAM for visual explanations
- [ ] Mobile app version (React Native or Flutter)
- [ ] Docker containerization for easy deployment
- [ ] Multi-damage detection (detect multiple damage types simultaneously)
- [ ] Integration with insurance claim systems
- [ ] Database support for storing predictions history

---

## 🚨 Limitations

- **Viewing Angle**: Best results with third-quarter views; other angles may reduce accuracy
- **Image Quality**: Requires clear, well-lit images for optimal performance
- **Training Data**: Limited to 6 specific damage categories
- **Single Damage**: Designed to classify primary damage type, not multiple simultaneous damages
- **Weather Conditions**: Performance may vary with different lighting and weather conditions

---

## 📈 Skills Demonstrated

**Deep Learning** · **Transfer Learning** · **ResNet50** · **EfficientNet** · **Custom CNN** · **PyTorch** · **Image Classification** · **Streamlit** · **FastAPI** · **REST API** · **Web Development** · **Computer Vision** · **Data Augmentation** · **Hyperparameter Tuning** · **Optuna** · **Model Deployment** · **Batch Normalization** · **Dropout** · **L2 Regularization** · **Insurance Tech** · **Automotive AI**

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Improve model accuracy with better architectures
- Add more damage categories
- Enhance the UI/UX
- Add comprehensive tests
- Improve documentation
- Optimize inference speed

---

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Reach out via email

---

## ⭐ Show Your Support

If you find this project helpful, please consider giving it a star ⭐ on GitHub!

---

