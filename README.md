# CMPE255-Project : Automated Skin Lesion Segmentation for Melanoma Detection using CRISP-DM Methodology

## Authors
Soham Raj Jain, Prachi Gupta

## Project Resources

| Resource | Link |
|----------|------|
| **Google Drive** | [Access Dataset & Files](https://drive.google.com/drive/folders/1cSEtOXLsFajT9ESzUVxmTvRjhv3d_cC3?usp=sharing) |
| **Presentation Deck** | [View Presentation](https://docs.google.com/presentation/d/1Gl6ffEALti2DKhBPrcjzvEeO0u4lxAH7eM7mu0FeNpc/edit?usp=sharing) |
| **Deployment** | [Live Demo](your-deployment-link-here) |
| **Explanation Video** |[Video](https://drive.google.com/file/d/1farHllhVM55IG8M0zZXznEQNwHZWgGbV/view?usp=sharing)  |

---
## CRISP-DM Methodology

This project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology, ensuring a structured and comprehensive approach to solving the melanoma detection challenge.

## Business Understanding

### Problem Statement
Automated skin lesion segmentation for melanoma detection represents a critical challenge in medical imaging and dermatology. Melanoma, the most dangerous form of skin cancer, requires early and accurate detection to ensure effective treatment outcomes.

### Business Impact
Early detection of melanoma is crucial for patient survival:
- **Early Detection Survival Rate**: 99%
- **Late Detection Survival Rate**: 27%

This dramatic difference underscores the life-saving potential of accurate and automated melanoma detection systems. By implementing computer vision techniques for skin lesion segmentation, we can:
- Assist dermatologists in making faster, more accurate diagnoses
- Reduce human error and subjectivity in lesion assessment
- Enable screening in underserved areas with limited access to specialists
- Process large volumes of dermoscopic images efficiently

### Project Goal
Develop a robust computer vision model that achieves:
- **Dice Coefficient**: > 0.89
- **Intersection over Union (IoU)**: > 0.83

These metrics ensure high-quality segmentation that accurately delineates lesion boundaries, providing reliable support for clinical decision-making.

---
## Data Understanding

### Dataset Overview
This project utilizes the **ISIC 2018 Challenge Dataset** (Task 1: Lesion Boundary Segmentation), which provides dermoscopic images of skin lesions along with their corresponding ground truth segmentation masks.

**Dataset Specifications:**
- **Total Training Images**: 2,594 dermoscopic images
- **Image Format**: RGB color images (.jpg)
- **Mask Format**: Binary segmentation masks
- **Source**: International Skin Imaging Collaboration (ISIC) Archive

### Dataset Characteristics

#### Image Dimensions
The dataset exhibits significant variability in image dimensions:
- **Unique image sizes**: 206 different dimension combinations
- **Size range**: From (566, 679) pixels to (4420, 6640) pixels
- This variability reflects real-world clinical imaging conditions where different devices and settings are used

#### Lesion Area Statistics
Analysis of the lesion coverage within images reveals:

| Metric | Value |
|--------|-------|
| **Mean Lesion Area** | 21.40% |
| **Median Lesion Area** | 13.81% |
| **Standard Deviation** | 20.83% |
| **Minimum Lesion Area** | 0.30% |
| **Maximum Lesion Area** | 98.66% |

![Lesion Area Distribution](exploration_output/statistics_20251207_111644.png)

**Key Observations:**
- The lesion area distribution is **right-skewed**, with most lesions occupying between 5% and 35% of the image
- The median (13.81%) is significantly lower than the mean (21.40%), indicating the presence of outliers with large lesion areas
- High standard deviation (20.83%) suggests substantial variability in lesion sizes across the dataset
- The presence of very small lesions (0.30%) and very large lesions (98.66%) presents a challenge for model training

### Sample Visualizations

Below are representative samples from the training dataset, showing the original dermoscopic images, their ground truth segmentation masks, and overlay visualizations:

![Training Samples](exploration_output/training_samples_20251207_111644.png)

**Sample Analysis:**

| Sample | Image ID | Dimensions | Lesion Area | Pixel Range |
|--------|----------|------------|-------------|-------------|
| 1 | ISIC_0000555 | (680, 833, 3) | 19.46% | [0, 233] |
| 2 | ISIC_0000112 | (1536, 2048, 3) | 20.27% | [0, 226] |
| 3 | ISIC_0011333 | (768, 1024, 3) | 34.59% | [17, 255] |
| 4 | ISIC_0010850 | (768, 1024, 3) | 13.63% | [0, 254] |

### Data Quality Observations

**Strengths:**
- High-quality dermoscopic images with clear lesion boundaries
- Expert-annotated ground truth masks ensuring accurate segmentation labels
- Diverse representation of lesion types, sizes, and imaging conditions
- Sufficient dataset size (2,594 images) for deep learning model training

**Challenges:**
- **Variable image dimensions** require preprocessing and standardization
- **Highly skewed lesion area distribution** may lead to class imbalance issues
- **Diverse lighting conditions and image quality** across different samples
- **Presence of artifacts** such as hair, rulers, and skin markings that may interfere with segmentation

### Data Preparation Requirements

Based on the exploration, the following preprocessing steps are necessary:
1. **Image resizing**: Standardize all images to a uniform dimension for model input
2. **Normalization**: Normalize pixel values to a consistent range
3. **Augmentation**: Apply data augmentation to handle class imbalance and improve model generalization
4. **Artifact handling**: Consider techniques to minimize the impact of hair and other artifacts
5. **Train-validation split**: Create stratified splits to ensure representative distribution across training and validation sets

---


# Data Preparation

## Preprocessing Pipeline

To address the challenges identified during data understanding, a comprehensive preprocessing pipeline was implemented:

### 1. Image Standardization
- **Resizing**: All images resized to 512×512 pixels to ensure uniform input dimensions
- **Normalization**: Pixel values normalized using ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Standard Deviation: [0.229, 0.224, 0.225]

### 2. Data Splitting Strategy
The dataset was split using a stratified approach to maintain representative distribution:

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training** | 2,075 | 80% |
| **Validation** | 260 | 10% |
| **Test** | 259 | 10% |

### 3. Data Augmentation

To improve model generalization and handle class imbalance, extensive augmentation techniques were applied to the training set:

**Geometric Transformations:**
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random rotation (±45°, p=0.5)
- Elastic transform (α=50, σ=10, p=0.3)
- Random resized crop (scale 0.8-1.0, p=0.5)

**Color Augmentations:**
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
- Gaussian blur (kernel 3-7, p=0.3)

**Rationale**: These augmentations simulate real-world variations in dermoscopic imaging, including different camera angles, lighting conditions, and image quality, thereby improving model robustness.

## Mask Preparation
- Binary thresholding applied to ground truth masks (threshold=127)
- Masks converted to single-channel format
- Perfect pixel-wise alignment maintained between images and masks

---

# Modeling

## Model Architecture: DeepLabV3+

DeepLabV3+ was selected as the segmentation architecture due to its superior performance in medical image segmentation tasks and ability to capture multi-scale contextual information.

### Architecture Overview

```
Input (512×512×3)
↓
Encoder (ResNet-50 Backbone)
├─ Layer 1 → Low-level features (256 channels)
├─ Layer 2, 3, 4 → High-level features (2048 channels)
↓
ASPP (Atrous Spatial Pyramid Pooling)
├─ 1×1 convolution
├─ 3×3 atrous conv (rate=6)
├─ 3×3 atrous conv (rate=12)
├─ 3×3 atrous conv (rate=18)
└─ Global Average Pooling
↓
Decoder
├─ Fuse with low-level features
├─ Depthwise separable convolutions
└─ Upsample to original resolution
↓
Output (512×512×1)
```

### Key Components

#### 1. Encoder (ResNet-50)
- Pre-trained on ImageNet for transfer learning
- Modified with atrous convolutions (output stride=16)
- Extracts multi-level features from input images

#### 2. ASPP Module
- Captures multi-scale contextual information
- Parallel atrous convolutions with rates [6, 12, 18]
- Global average pooling for image-level features
- Combines features from multiple receptive fields

#### 3. Decoder Module
- Recovers spatial information lost during encoding
- Fuses high-level semantic features with low-level spatial details
- Uses depthwise separable convolutions for efficiency
- Progressive upsampling to original resolution

### Model Specifications
- **Total Parameters**: ~41.26M
- **Trainable Parameters**: ~41.26M
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Output Stride**: 16

## Loss Function

A **Combined Loss** strategy was employed to leverage the strengths of multiple loss functions:

### Formula
```
Total Loss = 0.7 × Dice Loss + 0.3 × Focal Loss
```

### Components

#### 1. Dice Loss (70% weight)
- Optimizes for overlap between prediction and ground truth
- Directly related to the Dice coefficient metric
- Handles class imbalance effectively

#### 2. Focal Loss (30% weight)
- Addresses extreme class imbalance (lesion vs background)
- Parameters: α=0.25, γ=2.0
- Focuses learning on hard-to-classify pixels

**Rationale**: This combination ensures both high segmentation quality (Dice) and robust handling of difficult cases (Focal), leading to superior performance on the target metrics.

## Training Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **Optimizer** | AdamW | Decoupled weight decay for better generalization |
| **Learning Rate** | 0.001 | Balanced convergence speed and stability |
| **Weight Decay** | 0.0001 | L2 regularization to prevent overfitting |
| **Batch Size** | 12 | Maximum feasible on available GPU memory |
| **Epochs** | 30 (early stopped at 23) | Sufficient for convergence with early stopping |
| **Scheduler** | OneCycleLR | Dynamic learning rate for faster convergence |
| **Early Stopping** | Patience=10 | Prevents overfitting to training data |
| **Gradient Clipping** | Max norm=1.0 | Stabilizes training |

## Training Process

The model was trained using the following strategy:

1. **Initialization**: ResNet-50 backbone pre-trained on ImageNet
2. **Optimization**: AdamW optimizer with OneCycleLR scheduler
3. **Monitoring**: Validation Dice and IoU tracked every epoch
4. **Checkpointing**: Best model saved based on validation Dice score
5. **Early Stopping**: Training stopped after 10 epochs without improvement

### Training Hardware
- GPU: NVIDIA Tesla T4 / V100 (Google Colab)
- Training Time: ~2-3 hours for 23 epochs
- Memory Usage: ~10GB GPU memory

## Training Results

![Training History](checkpoints_deeplabv3plus/training_history.png)

### Key Observations

1. **Loss Convergence**: Both training and validation loss decreased steadily, indicating effective learning without overfitting

2. **IoU Performance**:
   - Started at ~0.64 (epoch 1)
   - Reached validation IoU of **0.855** (exceeding target of 0.83)
   - Validation IoU crossed target threshold around epoch 10

3. **Dice Performance**:
   - Started at ~0.78 (epoch 1)
   - Achieved validation Dice of **0.921** (exceeding target of 0.89)
   - Validation Dice crossed target threshold around epoch 8

4. **Early Stopping**: Training automatically stopped at epoch 23 due to no improvement in validation metrics for 10 consecutive epochs, preventing overfitting

5. **Generalization**: Small gap between training and validation metrics indicates good generalization without significant overfitting

---

# Evaluation

## Test Set Performance

The final model was evaluated on the held-out test set (259 images) using comprehensive metrics:

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Dice Coefficient** | **0.8907** | > 0.89 | ✅ **MET** |
| **Jaccard Index (IoU)** | **0.8206** | > 0.83 | ⚠️ **Near Target** |
| **Sensitivity (Recall)** | **0.8884** | - | - |
| **Specificity** | **0.9736** | - | - |
| **Precision** | **0.9222** | - | - |

## Performance Analysis

### ✅ Strengths

1. **Excellent Dice Score (0.8907)**
   - Achieved project target of > 0.89
   - Indicates high overlap between predicted and actual lesion boundaries
   - Demonstrates robust segmentation quality

2. **High Specificity (0.9736)**
   - Model correctly identifies non-lesion regions
   - Low false positive rate
   - Reduces unnecessary concern from healthy skin

3. **Strong Precision (0.9222)**
   - When model predicts lesion, it's correct 92% of the time
   - Builds trust in positive predictions

4. **Balanced Sensitivity (0.8884)**
   - Captures 88.84% of actual lesion pixels
   - Minimizes false negatives for critical medical application

### ⚠️ Areas for Improvement

1. **IoU Score (0.8206)**
   - Slightly below target of 0.83
   - Gap of ~0.01 points
   - Potential improvements through ensemble methods or longer training

## Qualitative Results

![Prediction Samples](checkpoints_deeplabv3plus/predictions_visualization.png)

### Sample-by-Sample Analysis

The visualization shows representative predictions across various lesion types:

- **Sample 1**: Large, irregularly shaped lesion - IoU: 0.880, Dice: 0.936
  - Excellent boundary delineation
  - Minimal over/under-segmentation

- **Sample 2**: Elongated lesion with complex boundaries - IoU: 0.923, Dice: 0.960
  - Outstanding performance on challenging shape
  - Precise edge detection

- **Sample 3**: Small, dark lesion with high contrast - IoU: 0.907, Dice: 0.951
  - Accurate segmentation despite small size
  - Clean boundary prediction

- **Sample 4**: Lesion with black frame artifacts - IoU: 0.904, Dice: 0.950
  - Robust to imaging artifacts
  - Successfully isolated lesion from frame

- **Sample 5**: Moderate-sized lesion with hair artifacts - IoU: 0.704, Dice: 0.826
  - More challenging case due to hair occlusion
  - Demonstrates model limitations on heavily occluded regions

### Common Success Patterns
- ✅ Accurate segmentation of well-defined lesion boundaries
- ✅ Robust performance across varying lesion sizes (0.3% - 98% of image)
- ✅ Handles color variation and different skin tones effectively
- ✅ Successfully segments lesions with irregular shapes

### Challenging Cases
- ⚠️ Hair artifacts can reduce segmentation accuracy
- ⚠️ Black frames and rulers in images occasionally cause minor boundary errors
- ⚠️ Very small lesions (<5% of image) may have slightly reduced IoU

## Comparison with Project Goals

| Objective | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Dice Coefficient | > 0.89 | 0.8907 | ✅ **EXCEEDED** |
| IoU (Jaccard Index) | > 0.83 | 0.8206 | ⚠️ 98.8% of target |
| Clinical Utility | High precision/sensitivity | 92%/89% | ✅ **EXCELLENT** |

**Overall Assessment**: The model successfully meets the primary project objective (Dice > 0.89) and comes very close to the IoU target, demonstrating strong clinical potential for automated melanoma detection support.
