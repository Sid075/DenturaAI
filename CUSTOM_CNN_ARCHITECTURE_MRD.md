# 🦷 Dentura AI — Technical Module Report (MRD)
## Custom 4-Block CNN Architecture, Scratch Baseline Engineering & Parameter Mathematics

**Author / Module Lead:** Custom CNN & Baseline Modeling Engineer  
**Project:** Dentura AI — Clinical Oral Pathology Platform  
**Target Audience:** Project Viva, Technical Evaluation Panel, and Deep Learning Reviewers  
**Document Status:** Complete & Verified  

---

## 1. Scope of Contributions Overview

In our 3-person engineering team, this module focuses on designing, calculating, and training the **scratch Convolutional Neural Network baseline**:

1. **Scratch Architecture Design**: Engineered a 4-block deep convolutional neural network from the ground up without using any pretrained ImageNet weights.
2. **Mathematical Parameter Optimization**: Kept the model compact at **457,670 total parameters (~457K)**, balancing deep feature representation with strict regularization to avoid overfitting on clinical samples.
3. **Global Average Pooling Innovation**: Eliminated dense feature flattening, saving over 12.8 million redundant parameters and reducing memory footprint to **1.75 MB**.
4. **Empirical Baseline Validation**: Achieved **75.78% test accuracy** and **75.19% weighted F1-score** across 1,115 held-out clinical test scans, proving the viability of compact scratch models and establishing the benchmark against which transfer learning was validated.

---

## 2. Deep Dive: Custom 4-Block CNN Pipeline

### 2.1 File Location & Architectural Context
- **Primary Source Code:** `notebooks/oral_pathology_training_and_benchmarks.ipynb` (Cell 14)
- **Model Checkpoint:** `cnn_model.h5`
- **Total Parameters:** 457,670 (Trainable: 456,710 | Non-trainable: 960)
- **Model Memory Footprint:** 1.75 MB

### 2.2 Why Build a Custom CNN from Scratch?
1. **Scientific Benchmark Requirement**: In medical computer vision, pre-trained transfer models (like EfficientNet or ResNet) cannot be evaluated in isolation. A custom scratch CNN proves whether pre-trained ImageNet features are actually necessary or if intraoral features can be learned natively.
2. **Computational Efficiency**: At 457K parameters, this architecture is over **50 times smaller** than ResNet50 (~23.6M params) and **9 times smaller** than EfficientNetB0 (~4.06M params), making it an ideal candidate for edge devices and low-resource dental clinic hardware.
3. **Architectural Control**: Designing from scratch allows exact control over receptive fields, downsampling factors, and regularization schemes.

---

### 2.3 Step-by-Step Code Explanation (`build_cnn`)

Here is the exact implementation code from Cell 14 of the notebook with a line-by-line technical breakdown:

```python
def build_cnn(input_shape=(224, 224, 3), num_classes=6):
    model = Sequential([
        # 1. Input Layer & Data Normalization
        Input(shape=input_shape),
        img_augmentation,
        layers.Rescaling(1./255),

        # 2. Convolutional Block 1 (Low-Level Feature Extraction)
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. Convolutional Block 2 (Mid-Level Texture & Edge Extraction)
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 4. Convolutional Block 3 (High-Level Pathology Pattern Extraction)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 5. Convolutional Block 4 (Complex Lesion & Structural Abstraction)
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 6. Feature Aggregation & Regularized Dense Classifier
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = optimizers.Adam(learning_rate=1e-3)
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
```

#### Line-by-Line Technical Walkthrough:

- **Lines 3–5 (`Input`, `img_augmentation`, `Rescaling`):**  
  Accepts intraoral RGB scans at `(224, 224, 3)`. The in-graph augmentation applies rotations ($\pm15\%$), translations ($\pm10\%$), horizontal flips, zoom ($\pm10\%$), and contrast jitter. `Rescaling(1./255)` converts 8-bit integers $[0, 255]$ into floating-point tensors in range $[0.0, 1.0]$.
- **Lines 7–10 (Block 1: Conv2D 32, 3x3 + BN + MaxPool):**  
  Extracts 32 basic edge, corner, and color-gradient filters. `padding='same'` preserves the 224x224 spatial resolution before $2 \times 2$ Max Pooling cuts spatial dimensions in half to `(112, 112, 32)`.
- **Lines 12–15 (Block 2: Conv2D 64, 3x3 + BN + MaxPool):**  
  Doubles feature channels to 64 to detect enamel demineralization textures and gingival contour changes. Downsamples spatial grid to `(56, 56, 64)`.
- **Lines 17–20 (Block 3: Conv2D 128, 3x3 + BN + MaxPool):**  
  Expands to 128 channels for complex regional patterns like calculus accretion and tooth discoloration stains. Downsamples spatial grid to `(28, 28, 128)`.
- **Lines 22–25 (Block 4: Conv2D 256, 3x3 + BN + MaxPool):**  
  Captures 256 high-level semantic representations of ulcerations, missing tooth gaps (hypodontia), and deep cavitations. Outputs a compact tensor of `(14, 14, 256)`.
- **Line 27 (`GlobalAveragePooling2D`):**  
  Averages each $14 \times 14$ feature map into a single scalar value. This collapses the `(14, 14, 256)` tensor into a 1D vector of length `256` with **zero parameters**, entirely avoiding the parameter explosion of traditional flattening.
- **Lines 28–30 (`Dense(256)`, `Dropout(0.5)`, `Dense(6, 'softmax')`):**  
  Projects the 256-dimensional pooled vector into a 256-unit fully connected latent space with ReLU activation. 50% Dropout eliminates co-adaptation of features. The final Dense layer produces normalized softmax probabilities across the 6 dental disease classes.

---

## 3. Mathematical Parameter Calculations (Crucial for Viva)

Examiners often ask: *"Show how you calculated the parameter count for each layer."*

### 3.1 Convolutional Layer Parameter Formula
$$\text{Parameters} = \left[(\text{Kernel Width} \times \text{Kernel Height} \times \text{Input Channels}) + 1_{\text{bias}}\right] \times \text{Output Filters}$$

- **Conv Block 1:**  
  $[(3 \times 3 \times 3) + 1] \times 32 = (27 + 1) \times 32 = 28 \times 32 = \mathbf{896}$
- **Conv Block 2:**  
  $[(3 \times 3 \times 32) + 1] \times 64 = (288 + 1) \times 64 = 289 \times 64 = \mathbf{18,496}$
- **Conv Block 3:**  
  $[(3 \times 3 \times 64) + 1] \times 128 = (576 + 1) \times 128 = 577 \times 128 = \mathbf{73,856}$
- **Conv Block 4:**  
  $[(3 \times 3 \times 128) + 1] \times 256 = (1152 + 1) \times 256 = 1153 \times 256 = \mathbf{295,168}$

### 3.2 Batch Normalization Parameter Formula
Each channel maintains 4 parameters: Gamma ($\gamma$, scale), Beta ($\beta$, offset), Moving Mean ($\mu$), and Moving Variance ($\sigma^2$):
$$\text{Parameters} = 4 \times \text{Number of Channels}$$
- **BN 1:** $4 \times 32 = \mathbf{128}$
- **BN 2:** $4 \times 64 = \mathbf{256}$
- **BN 3:** $4 \times 128 = \mathbf{512}$
- **BN 4:** $4 \times 256 = \mathbf{1,024}$  
*(Note: 2 parameters per channel are trainable [$\gamma, \beta$], and 2 are non-trainable moving statistics [$\mu, \sigma^2$]. Total non-trainable across model = 960).*

### 3.3 Dense Layer Parameter Formula
$$\text{Parameters} = (\text{Input Dimensions} + 1_{\text{bias}}) \times \text{Output Neurons}$$
- **Dense Layer 1:**  
  $(256 + 1) \times 256 = 257 \times 256 = \mathbf{65,792}$
- **Output Softmax Layer:**  
  $(256 + 1) \times 6 = 257 \times 6 = \mathbf{1,542}$

### 3.4 Grand Total Verification
$$\text{Total} = 896 + 128 + 18,496 + 256 + 73,856 + 512 + 295,168 + 1,024 + 65,792 + 1,542 = \mathbf{457,670}$$

---

## 4. Key Architectural Decisions Explained

### 4.1 Why 3x3 Kernels Across All Blocks?
Stacking two $3 \times 3$ convolutional layers gives an effective receptive field of $5 \times 5$; stacking three gives $7 \times 7$. However:
- A $7 \times 7$ kernel requires $7 \times 7 = 49$ weights per channel.
- Three stacked $3 \times 3$ kernels require $3 \times (3 \times 3) = 27$ weights—a **45% parameter reduction**.
- Stacking $3 \times 3$ convolutions introduces 3 consecutive ReLU non-linearities instead of 1, allowing the network to learn more discriminative decision boundaries.

### 4.2 Why Global Average Pooling (GAP) vs Flatten?
This is the most critical design choice of the Custom CNN:
- **Traditional Flatten Approach (VGG-style):**  
  $14 \times 14 \times 256 = 50,176$ dimensional vector.  
  A single Dense(256) layer would require: $(50,176 + 1) \times 256 = \mathbf{12,845,312\text{ parameters}}$!
- **Global Average Pooling Approach (Our Design):**  
  Averages each $14 \times 14$ slice down to its mean value $\rightarrow$ 256 vector.  
  Parameters required: **0 parameters**.
- **Clinical Benefit:** Prevents the network from memorizing training photos, enforces translational invariance, and keeps the model lightweight enough to train stably from scratch on 4,373 images.

---

## 5. Empirical Performance & Training Dynamics

### 5.1 Test Performance on 1,115 Held-Out Clinical Scans
- **Test Accuracy:** **75.78%** (845 / 1,115 correct predictions)
- **Balanced Accuracy:** **65.03%**
- **Macro Precision / Recall / F1:** 73.83% / 65.03% / **66.65%**
- **Weighted Precision / Recall / F1:** 76.03% / 75.78% / **75.19%**

### 5.2 Per-Class Breakdown (Where Custom CNN Excelled & Struggled)
- **Hypodontia (Missing Teeth):** F1 = **95.0%** (238 / 251 correct) — Strong structural pattern detection.
- **Gingivitis (Gum Inflammation):** F1 = **80.0%** (353 / 470 correct) — Good color/texture differentiation.
- **Calculus (Tartar):** F1 = **77.0%** (182 / 260 correct) — Confused with Gingivitis (75 instances).
- **Mouth Ulcer:** F1 = **75.0%** (48 / 53 correct) — High recall for mucosal lesions.
- **Caries & Discoloration:** Lower F1 (34% and 35%) due to class imbalance in training set (only 219 and 183 original images).

### 5.3 Comparison Against Benchmark Models

| Model | Architecture | Parameters | Test Accuracy | Weighted F1 | Role |
|---|---|:---:|:---:|:---:|---|
| **Custom 4-Block CNN** | Scratch Conv2D + GAP | **~457K** | **75.78%** | **75.19%** | **Scratch Baseline** |
| **ResNet50** | Transfer Learning (ImageNet) | ~23.6M | 73.18% | 72.59% | Transfer Baseline (Overfitted) |
| **EfficientNetB0** | Transfer Learning + MBConv | ~4.06M | **82.51%** | **82.52%** | **Champion Deployed Model** |

**Key Finding:** The Custom CNN (75.78%) actually **outperformed ResNet50 (73.18%)** by 2.6% despite having 50x fewer parameters, because ResNet50's 23M parameters overfitted on the minority pathology classes.

---

## 6. Complete Viva Speaking Script (2–3 Minutes)

> *"Good morning professors. My responsibility in this project was engineering our **scratch baseline model: the Custom 4-Block Convolutional Neural Network**.*
>
> *Before applying transfer learning, we needed an empirical baseline to determine whether intraoral diseases could be classified without pre-trained ImageNet weights. I architected a hierarchical 4-block Conv2D network with progressive filter expansion from 32 to 64, 128, and 256 channels.*
>
> *Every convolutional layer uses standard 3x3 kernels with 'same' padding to preserve spatial context, coupled with Batch Normalization for internal gradient stability and 2x2 Max Pooling to downsample the 224x224 image through 112, 56, 28, and 14 pixels.*
>
> *A major innovation in this architecture is the use of **Global Average Pooling** instead of traditional dense flattening. Had we flattened the 14x14x256 feature map, a single fully connected layer would have exploded to over 12.8 million parameters, leading to severe overfitting. Global Average Pooling reduced the spatial grid to 256 features with zero parameters, keeping the entire model ultra-compact at just **457,670 parameters**.*
>
> *With 50% Dropout regularization and the Adam optimizer, our Custom CNN achieved **75.78% test accuracy and 75.19% weighted F1-score** across 1,115 test images. Notably, it outperformed our 23-million parameter ResNet50 baseline, proving that an appropriately regularized, compact network is superior to an over-parameterized model on specialized medical imaging."*

---

## 7. Expected Viva Questions & Perfect Answers

#### Q1: "Why did you double the number of filters from 32 to 64, 128, and 256 at each block?"
**Answer:**  
*"As the spatial resolution decreases through Max Pooling (224 $\rightarrow$ 112 $\rightarrow$ 56 $\rightarrow$ 28 $\rightarrow$ 14), the network needs to capture richer, more complex semantic combinations. Doubling the filters compensates for the spatial reduction, ensuring total information capacity is preserved while shifting the representation from simple edges to complex lesion textures."*

#### Q2: "How does Global Average Pooling prevent overfitting?"
**Answer:**  
*"Traditional flattening retains spatial position weights, which memorizes exact lesion locations from the training set. Global Average Pooling computes the average activation of each feature map across the entire $14 \times 14$ grid. This forces the conv layers to act as global feature detectors (e.g., 'is calculus present anywhere in this map?') rather than memorizing spatial coordinates, and eliminates 12.8 million dense weights."*

#### Q3: "What is the purpose of Batch Normalization after every Conv2D layer?"
**Answer:**  
*"Batch Normalization standardizes the mean and variance of each mini-batch feature map. This reduces internal covariate shift, allowing us to use a higher learning rate ($10^{-3}$) without gradient explosion, speeds up convergence, and provides a slight regularizing effect."*

#### Q4: "Why did your 457K Custom CNN perform better than ResNet50 with 23M parameters?"
**Answer:**  
*"ResNet50 has massive representational capacity. With only 4,373 training images and severe class imbalance (e.g., Caries has only 219 images), ResNet50 overfitted on the majority classes. Our Custom CNN was right-sized for this dataset volume, using Global Average Pooling, Batch Normalization, and 50% Dropout to generalize effectively."*
