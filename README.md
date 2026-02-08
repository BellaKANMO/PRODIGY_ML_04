# PRODIGY_ML_04
# This project was carried out as part of my internship at Prodigy InfoTech as a Machine Learning intern

## Hand Gesture Recognition with Deep Learning

### Project Overview
This project implements a **hand gesture recognition system** using **Convolutional Neural Networks (CNN)** and **computer vision techniques**.  
The goal is to accurately identify and classify 10 different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.  
This work demonstrates the practical application of **deep learning for real-time gesture recognition**.

---

### Objectives
- Develop a robust CNN model for hand gesture classification
- Process and prepare image data from the LeapGestRecog dataset
- Achieve high accuracy in recognizing 10 distinct hand gestures
- Implement real-time gesture recognition using webcam
- Enable smooth predictions with temporal filtering

---

### Dataset
The project uses the **LeapGestRecog dataset** from Kaggle, containing:
- **20,000 images** of hand gestures (2,000 per class)
- **10 gesture classes**: Palm, L, Fist, Fist Moved, Thumb, Index, OK, Palm Moved, C, Down
- Images captured from different people and angles
- Grayscale images resized to 64x64 pixels

**Dataset Source**: [LeapGestRecog on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

---

### Methodology

1. **Data Loading & Exploration**
   - Loading images from multiple gesture folders
   - Converting images to grayscale
   - Resizing all images to 64x64 pixels
   - Analyzing class distribution

2. **Data Preprocessing**
   - Train/test split (80/20) with stratification
   - Normalization of pixel values (0-1 range)
   - Conversion to PyTorch tensors
   - DataLoader creation with batch size of 32

3. **Model Architecture**
   - Custom CNN architecture with:
     - 2 convolutional layers (32 and 64 filters)
     - MaxPooling layers
     - Dropout for regularization
     - Fully connected layers
     - Softmax output for 10 classes

4. **Training & Optimization**
   - Loss function: CrossEntropyLoss
   - Optimizer: Adam
   - Multiple training epochs
   - Model checkpointing

5. **Evaluation & Real-time Prediction**
   - Classification metrics (accuracy, precision, recall)
   - Confusion matrix visualization
   - Real-time webcam integration using MediaPipe
   - Temporal smoothing with prediction buffer

---

### Technologies Used
- **Python** - Core programming language
- **PyTorch** - Deep learning framework
- **OpenCV (cv2)** - Computer vision and image processing
- **MediaPipe** - Hand landmark detection
- **NumPy** - Numerical computations
- **Scikit-learn** - Train/test split and metrics
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Development environment

---

### Results
The model successfully achieves:
- **High classification accuracy** on the test set
- **Real-time gesture recognition** from webcam feed
- **Robust hand detection** using MediaPipe landmarks
- **Smooth predictions** through temporal filtering
- Clear visualization of detected gestures with confidence scores

The system can recognize:
- Static gestures (Palm, Fist, OK, Thumb, Index, L, C, Down)
- Dynamic gestures (Fist Moved, Palm Moved)

These capabilities enable applications in:
- **Human-computer interaction**
- **Contactless control systems**
- **Sign language recognition**
- **Gaming interfaces**
- **Accessibility tools**

---

### How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/PRODIGY_ML_04.git
   cd PRODIGY_ML_04
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Download the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
   - Extract it to the project directory as `leapGestRecog/`

4. **Run the notebook**:
   - Open `hand_gesture.ipynb` in Jupyter Notebook
   - Run cells sequentially to train the model
   - The trained model will be saved as `hand_gesture_pytorch.pth`

5. **Real-time gesture recognition**:
   - Execute the final cells to start webcam recognition
   - Show your hand gestures to the camera
   - Press 'q' to quit

---

### Project Structure
```
PRODIGY_ML_04/
│
├── hand_gesture.ipynb          # Main Jupyter notebook
├── hand_gesture_pytorch.pth    # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── leapGestRecog/             # Dataset folder (to be downloaded)
```

---

### Key Features
- **CNN-based classification** with custom architecture
- **MediaPipe integration** for robust hand detection
- **Real-time processing** with webcam input
- **Temporal smoothing** for stable predictions
- **Visual feedback** with bounding boxes and confidence scores
- **Well-documented code** with clear explanations

---

### Notes
- The project uses PyTorch for deep learning implementation
- MediaPipe is used for hand detection and landmark extraction
- The model processes grayscale 64x64 images for efficiency
- Prediction smoothing uses a 5-frame buffer for stability
- All steps are reproducible within the notebook
- No external utilities or complex scripts required

---

### Future Improvements
- Expand the dataset with more diverse hand poses
- Implement data augmentation for better generalization
- Experiment with deeper CNN architectures
- Add support for multi-hand gesture recognition
- Develop a standalone application with GUI
- Optimize model for mobile deployment

---

### Acknowledgments
- Dataset provided by the GTI-UPM team on Kaggle
- Project completed during internship at **Prodigy InfoTech**
- Built with open-source libraries and frameworks