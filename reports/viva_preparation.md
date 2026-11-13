# FOODWISE AI: Technical Viva Voce Preparation Guide (30 Questions & Detailed Answers)

**Project:** AI-Based Food Waste Detection and Categorization System Using YOLOv11 Instance Segmentation  
**Author:** Senior Machine Learning & Computer Vision Research Team  
**Scope:** Convolutional Neural Networks, Deep Learning, YOLOv11 Instance Segmentation, Mathematical Formulations, Model Evaluation, Sustainability & SDG 12  

---

### Section 1: Deep Learning Fundamentals & Convolutional Neural Networks (Q1–Q8)

#### Q1: What is a Convolutional Neural Network (CNN) and why is it preferred over Fully Connected Networks for image processing?
**Answer:**  
A Convolutional Neural Network (CNN) is a specialized deep neural architecture designed for grid-structured data such as 2D images. Unlike Fully Connected (Dense) networks where every input pixel connects to every neuron—resulting in massive parameter counts and loss of spatial geometry—CNNs leverage three core inductive biases:
1. **Local Connectivity (Receptive Fields):** Neurons compute convolutions over localized spatial neighborhoods ($3\times3$ or $5\times5$), capturing local spatial correlations.
2. **Weight Sharing:** The same filter kernel slides across the entire image tensor, dramatically reducing trainable parameters and conferring translation equivariance:
   $$(I * K)(i, j) = \sum_m \sum_n I(i - m, j - n) K(m, n)$$
3. **Spatial Hierarchies:** Stacking convolutional layers interspersed with pooling or strided operations forms hierarchical representations: shallow layers capture low-level features (edges, textures, colors), while deeper layers represent high-level semantic abstractions (food shapes, cutlery contours, plate boundaries).

#### Q2: What is the Vanishing Gradient problem and how do modern CNN architectures mitigate it?
**Answer:**  
In deep neural networks trained with backpropagation, gradients are computed via the chain rule of calculus:
$$\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial a_L} \prod_{k=l+1}^L \frac{\partial a_k}{\partial a_{k-1}} \frac{\partial a_l}{\partial W_l}$$
When using saturating activation functions like $\sigma(z)$ (whose derivative $\sigma'(z) \le 0.25$) or weights initialized with small values, the continuous multiplication of terms $< 1$ causes the gradient to decay exponentially toward zero as it propagates back to the initial layers. Consequently, early layers stop learning.

Modern CNNs mitigate this through:
1. **Non-saturating Activations:** Utilizing ReLU ($\max(0, x)$), Leaky ReLU, or SiLU/Swish ($x \cdot \sigma(x)$), which possess non-zero constant or unit derivatives for positive inputs.
2. **Residual Skip Connections (He et al., ResNet):** Identity shortcuts formulated as $y = \mathcal{F}(x, \{W_i\}) + x$ ensure that $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + I \right)$, guaranteeing that even if the gradient of the learned path $\frac{\partial \mathcal{F}}{\partial x}$ vanishes, the identity term $I$ allows gradients to flow unimpeded directly to early layers.
3. **Batch and Layer Normalization:** Normalizing layer activations to zero mean and unit variance stabilizes forward and backward distributions.

#### Q3: Explain the role of Receptive Field in CNNs and how dilated convolutions or SPPF expand it.
**Answer:**  
The receptive field of a neuron is the spatial subregion of the original input image that influences that neuron's activation. As network depth increases, the effective receptive field expands linearly with kernel size:
$$RF_{l} = RF_{l-1} + (k_l - 1) \cdot S_{l-1}$$
where $k$ is kernel size and $S$ is the cumulative stride product.
- **Dilated (Atrous) Convolutions:** Insert spaces (zeros) between kernel elements governed by a dilation rate $r$, enlarging the receptive field to $k' = k + (k - 1)(r - 1)$ without increasing parameters or downsampling resolution.
- **Spatial Pyramid Pooling - Fast (SPPF):** Passes feature maps sequentially through multiple small pooling layers (e.g. three consecutive $5\times5$ max-pooling operations) and concatenates their outputs. This efficiently captures multi-scale spatial context equivalent to large pooling operations ($5\times5$, $9\times9$, $13\times13$) at drastically reduced computational overhead.

#### Q4: Why is Transfer Learning vital in deep learning computer vision projects?
**Answer:**  
Transfer learning involves taking a neural network pretrained on a vast, diverse general visual corpus (such as ImageNet with 1.2M images or COCO with 80 classes) and fine-tuning its weights on a task-specific target dataset (such as canteen food waste).  
It provides three fundamental advantages:
1. **Feature Reusability:** The early layers already compute robust, universal representations (Gabor-like edge detectors, chromatic contrasts, textured contours) eliminating the need to learn these fundamentals from scratch.
2. **Data Efficiency & Regularization:** Pretrained models converge with far smaller domain-specific datasets (e.g. hundreds of annotated plate images) without severe overfitting.
3. **Accelerated Convergence:** The model starts in a parameter space close to a low-loss basin, reaching optimal performance in tens of epochs rather than hundreds.

#### Q5: What is the difference between Semantic Segmentation, Object Detection, and Instance Segmentation?
**Answer:**  
- **Object Detection:** Identifies the presence and category of objects and localizes them with rectangular bounding boxes coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$. It does not distinguish exact irregular object boundaries.
- **Semantic Segmentation:** Classifies every individual pixel in the image into a semantic class label (e.g., all rice pixels are labeled "rice"). However, it does not distinguish between distinct individual instances of the same class (two adjacent chicken breasts are merged into a single homogeneous semantic blob).
- **Instance Segmentation:** Simultaneously solves detection and pixel-level delineations: it detects each individual object instance, assigns its class label, and produces a unique binary pixel mask for every separate instance (distinguishing Chicken #1 from Chicken #2).

#### Q6: Explain Data Augmentation in the context of instance segmentation. Why must augmentations be synchronous?
**Answer:**  
Data augmentation synthetically introduces geometric and photometric variations into the training pipeline to prevent memorization and improve generalizability against real-world domain shifts (e.g., varying lighting, camera angles, food arrangements).  
In **Instance Segmentation**, every geometric transformation (horizontal flipping, random rotation, scaling, translation, shear) applied to the input image matrix $I$ must be **synchronously** applied to the target polygon vertex coordinates $P = \{(x_i, y_i)\}$ using the identical affine transformation matrix $A$:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = A \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
If image transforms and polygon transforms become desynchronized, ground-truth masks will label empty plate background, causing catastrophic gradient divergence.

#### Q7: What are the trade-offs between two-stage instance segmentation (Mask R-CNN) and single-stage instance segmentation (YOLOv11-seg)?
**Answer:**  
- **Mask R-CNN (Two-Stage):**
  1. *Stage 1:* Region Proposal Network (RPN) generates candidate regions of interest (RoIs).
  2. *Stage 2:* RoIAlign pools feature patches, followed by parallel branches for bounding box regression, classification, and FCN binary mask prediction.  
  *Advantage:* High spatial precision on small objects.  
  *Disadvantage:* Extremely slow inference (5–12 FPS), computationally expensive, high memory footprint, unsuited for real-time edge cafeteria stations.
- **YOLOv11-seg (Single-Stage):**
  Predicts bounding boxes, class logits, and dynamic mask coefficients in a single forward pass while a parallel Proto head outputs high-resolution mask prototypes across the whole image.  
  *Advantage:* Real-time inference speed (60–120 FPS), low parameter count (2.8M for YOLO11n), low latency, and unified multi-task loss optimization.

#### Q8: What is Overfitting and how do we detect and combat it in instance segmentation?
**Answer:**  
Overfitting occurs when a high-capacity model learns high-frequency noise and idiosyncrasies of the training images rather than generalizable underlying distributions.  
- **Detection:** Monitored via the divergence between Training Loss and Validation Loss: training loss continues to decrease while validation loss increases or validation mAP plateaus.
- **Countermeasures:**
  1. Stochastic regularization: Dropout, DropBlock, and Weight Decay ($L_2$ regularization $\frac{1}{2}\lambda \|W\|_2^2$).
  2. Synchronous geometric and photometric data augmentations (Mosaic, Copy-Paste, HSV jitter).
  3. Early stopping based on validation mAP@50-95.
  4. Using compact parameter architectures (YOLO11n-seg with 2.87M parameters).

---

### Section 2: YOLOv11 Architecture & Instance Segmentation Mechanisms (Q9–Q16)

#### Q9: Describe the overall architecture of YOLOv11 for instance segmentation.
**Answer:**  
YOLOv11-seg is structured into three primary sub-networks:
1. **Backbone:** Deep convolutional network built with **C3k2** blocks (optimized cross-stage partial layers with $2\times$ kernel processing) and **SPPF**, terminating in **C2PSA** (Cross-Stage Partial with Spatial Attention). It outputs feature hierarchies at three spatial scales ($P3$: stride 8, $P4$: stride 16, $P5$: stride 32).
2. **Neck (PANet / FPN):** Path Aggregation Network combining top-down and bottom-up pathways to fuse high-level semantic abstractions with low-level spatial detail.
3. **Head:** Decoupled detection heads that independently predict class probabilities and bounding box offsets via Distribution Focal Loss (DFL), operating in parallel with the **Proto Head** which outputs $k=32$ prototype mask channels.

#### Q10: How does YOLOv11 generate instance segmentation masks from prototype masks?
**Answer:**  
YOLOv11 employs the YOLACT/YOLO-seg prototype-based masking mechanism:
1. The **Proto Head** outputs a set of $k$ image-sized prototype masks $P \in \mathbb{R}^{k \times \frac{H}{4} \times \frac{W}{4}}$ that capture basis visual primitives across the image.
2. For each detected bounding box instance $i$, the detection head predicts $k$ scalar mask coefficients $C_i \in \mathbb{R}^k$.
3. The instance mask is generated via a matrix multiplication (linear combination) between the prototype tensors and the coefficient vector, passed through an element-wise sigmoid activation $\sigma$, and cropped to the predicted bounding box $B_i$:
   $$M_i = \sigma(C_i \cdot P) \odot \text{Crop}(B_i)$$
This decoupled prototype design enables real-time mask generation with minimal computational overhead.

#### Q11: Explain the purpose and operation of C2PSA (Cross-Stage Partial with Spatial Attention) in YOLOv11.
**Answer:**  
C2PSA integrates a spatial self-attention mechanism within the Cross-Stage Partial architecture.  
In dense food scenes where different food items touch, overlap, or blend (e.g. rice grains resting on top of chicken curry), local convolutions struggle to separate boundaries. C2PSA computes scaled dot-product spatial attention across the downsampled feature tensor:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
This permits each spatial position to attend to globally relevant visual cues across the plate, isolating distinct item contours before masks are generated.

#### Q12: What is the Multi-Task Loss function in YOLOv11-seg?
**Answer:**  
The total training loss $\mathcal{L}_{\text{total}}$ is a linear combination of four specialized loss objectives:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}$$
- $\mathcal{L}_{\text{box}}$: Complete IoU (CIoU) loss measuring bounding box overlap, center distance, and aspect ratio consistency.
- $\mathcal{L}_{\text{cls}}$: Binary Cross-Entropy (BCE) loss over class logits.
- $\mathcal{L}_{\text{dfl}}$: Distribution Focal Loss representing box coordinates as continuous probability distributions rather than Dirac delta points.
- $\mathcal{L}_{\text{seg}}$: Binary Cross-Entropy computed at the pixel level between the predicted instance mask raster and the ground-truth binary mask within the ground-truth bounding box:
  $$\mathcal{L}_{\text{seg}} = -\frac{1}{N}\sum_{p \in B} \left[ y_p \log \hat{y}_p + (1 - y_p) \log (1 - \hat{y}_p) \right]$$

#### Q13: What is Non-Maximum Suppression (NMS) and how is it applied in instance segmentation?
**Answer:**  
During inference, the dense anchor-free head outputs thousands of candidate bounding boxes with high overlap for the same physical object.  
**NMS Algorithm:**
1. Filter out all predictions whose classification confidence $s_i < \tau_{\text{conf}}$.
2. Sort remaining candidate boxes in descending order of confidence score.
3. Select the highest-scoring candidate $B_{\text{top}}$ and add it to the final output list.
4. Compute the Intersection over Union ($\text{IoU}$) between $B_{\text{top}}$ and all remaining candidate boxes.
5. Discard any box $B_j$ where $\text{IoU}(B_{\text{top}}, B_j) \ge \tau_{\text{IoU}}$ (typically $0.45$–$0.70$).
6. Repeat until all candidates are processed.  
In instance segmentation, NMS operates on the bounding boxes first; only the surviving boxes have their corresponding prototype mask coefficients assembled into final polygonal masks.

#### Q14: What is Distribution Focal Loss (DFL) and why does YOLOv11 use it instead of direct coordinate regression?
**Answer:**  
Direct coordinate regression (e.g. predicting $x, y, w, h$ with Smooth $L_1$ loss) treats object boundaries as unambiguous sharp edges. In real-world food images, food edges (such as rice mounds, poured soup, or grated cheese) are naturally blurred, irregular, or occluded.  
DFL models the boundary offsets as continuous probability distributions over a discrete set of bins $\{y_0, y_1, \dots, y_n\}$:
$$\hat{y} = \sum_{i=0}^n \text{softmax}(S_i) \cdot y_i$$
This enables the network to express uncertainty along ambiguous food contours, leading to significantly tighter bounding boxes and higher localization precision.

#### Q15: How does Mosaic Augmentation work and why is it beneficial for food waste detection?
**Answer:**  
Mosaic augmentation stitches four random training photographs together into a single composite $2\times2$ image tensor:
1. Four images are randomly cropped, scaled, and placed in quadrants around a movable center point.
2. Their corresponding bounding boxes and polygon masks are translated and clipped to the composite frame.  
*Benefits for Canteen Waste Detection:*
- Forces the model to recognize objects in unconventional contexts and at significantly smaller scales.
- Provides multi-object scenes per mini-batch, improving batch normalization statistics even with small batch sizes ($B=4$).
- Simulates crowded plate setups where multiple foods crowd into tight compartments.

#### Q16: Why is YOLOv11 better suited for plate waste detection than the custom U-Net model used in earlier iterations?
**Answer:**  
1. **Instance Awareness:** U-Net performs purely semantic segmentation—it classifies pixels without distinguishing individual object instances. It cannot differentiate two chicken pieces or associate a specific piece of cutlery with a plate.
2. **Context & Scale:** U-Net lacks spatial attention and multi-scale feature pyramids (FPN/PANet), resulting in fragmented, noisy mask blobs on irregular foods.
3. **Multi-Class Capacity:** U-Net struggles to scale effectively to 85 fine-grained classes with high class imbalance, whereas YOLOv11 utilizes decoupled heads, focal losses, and multi-scale feature fusion designed specifically for complex multi-class scenarios.

---

### Section 3: Evaluation Metrics & Mathematical Rigor (Q17–Q22)

#### Q17: Define Intersection over Union (IoU) mathematically and explain its geometric interpretation.
**Answer:**  
Intersection over Union (IoU), or the Jaccard Index, measures the degree of spatial overlap between a predicted region $A$ and a ground-truth region $B$:
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$
- In **Bounding Box Detection**, $A$ and $B$ are continuous rectangular geometries:
  $$\text{IoU}_{\text{box}} = \frac{\text{Area of Overlap Rectangle}}{\text{Area of Union Polygon}}$$
- In **Instance Segmentation**, $A$ and $B$ are discrete sets of binary pixels:
  $$\text{IoU}_{\text{mask}} = \frac{\sum_{i,j} (M_{\text{pred}}[i,j] \land M_{\text{gt}}[i,j])}{\sum_{i,j} (M_{\text{pred}}[i,j] \lor M_{\text{gt}}[i,j])}$$
IoU ranges strictly from $0.0$ (no spatial overlap) to $1.0$ (perfect spatial congruence).

#### Q18: Distinguish between Precision, Recall, and F1-Score in object detection and instance segmentation.
**Answer:**  
Given a chosen IoU threshold (e.g. $\text{IoU} \ge 0.50$):
- **True Positive (TP):** A prediction that correctly matches a ground-truth object with $\text{IoU} \ge \tau$ and correct class label.
- **False Positive (FP):** A prediction with $\text{IoU} < \tau$ or assigned to a non-existent object / wrong class.
- **False Negative (FN):** A ground-truth object that was missed by the model.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{Correct Predictions}}{\text{Total Predictions Made}}$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{Correct Predictions}}{\text{Total Real Objects Present}}$$
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **High Precision, Low Recall:** The model is conservative; when it predicts rice, it is almost certainly rice, but it misses many actual servings.
- **Low Precision, High Recall:** The model detects almost all food items, but produces numerous false alarms on napkins or plate edges.
- **F1-Score:** Harmonic mean balancing both metrics.

#### Q19: Explain the mathematical calculation of Average Precision (AP) and mean Average Precision (mAP).
**Answer:**  
For a specific class, predictions are sorted in descending order of confidence score. As the confidence threshold is lowered from $1.0$ down to $0.0$, precision and recall are computed cumulatively at each step, tracing the Precision-Recall (PR) curve $p(r)$.  
1. **Average Precision (AP):** Under the continuous VOC/COCO definition, AP is the area under the PR curve using all-point interpolation to ensure monotonic decay:
   $$\text{AP} = \int_{0}^{1} p_{\text{interp}}(r) \, dr \quad \text{where} \quad p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$
2. **Mean Average Precision (mAP):** The arithmetic mean of AP scores across all $C$ target classes:
   $$\text{mAP} = \frac{1}{C}\sum_{c=1}^C \text{AP}_c$$
- **mAP@50:** mAP calculated at a fixed IoU threshold of $0.50$.
- **mAP@50-95:** The primary COCO metric, which computes the average of mAP across 10 IoU thresholds from $0.50$ to $0.95$ in increments of $0.05$:
  $$\text{mAP@[.50:.95]} = \frac{1}{10} \sum_{k=0}^9 \text{mAP}_{@(\text{IoU} = 0.50 + 0.05k)}$$

#### Q20: Why is reporting Detection Confidence as Precision scientifically incorrect?
**Answer:**  
- **Confidence Score $s \in [0, 1]$:** Represents the model's internal posterior probability estimate $P(\text{class} \mid x, \text{box})$ that a detected bounding box belongs to a certain class. It is an uncalibrated output of a single prediction.
- **Precision $\in [0, 1]$:** An objective statistical measure evaluated across an entire ground-truth dataset comparing True Positives against all Positives at a specific IoU threshold.  
Confusing confidence with precision is a severe methodological flaw: an overconfident, uncalibrated model can output $0.98$ confidence on a completely false prediction (FP), yielding a precision of $0.0$. Precision can only be derived by auditing predictions against ground-truth annotations.

#### Q21: What is a Confusion Matrix in multi-class instance segmentation and what does its background class represent?
**Answer:**  
A confusion matrix is a $K \times K$ table (or $(K+1) \times (K+1)$ with background) where rows represent true ground-truth classes and columns represent model predictions:
- Matrix entry $C_{i,j}$ counts how many times ground-truth class $i$ was predicted as class $j$.
- **Diagonal elements $C_{i,i}$:** Correct predictions (True Positives).
- **Off-diagonal elements $C_{i,j}$ ($i \ne j$):** Misclassifications (e.g., `chicken` confused with `pork loin`).
- **Background Row/Column:**
  - Ground-truth class $i$ predicted as Background: **False Negative (missed object)**.
  - Background predicted as Class $j$: **False Positive (spurious detection on empty plate)**.

#### Q22: What are the differences between Box mAP and Mask mAP?
**Answer:**  
- **Box mAP:** Evaluates localization accuracy based on rectangular bounding box IoU ($\text{IoU}_{\text{box}} \ge \tau$). Because rectangles enclose empty background corners around organic food shapes, Box mAP can be artificially inflated even if the food contour is poorly localized.
- **Mask mAP:** Evaluates true pixel-level segmentation accuracy ($\text{IoU}_{\text{mask}} \ge \tau$). It requires exact boundary adherence. Mask mAP is virtually always lower than Box mAP because matching irregular organic food boundaries at high IoU thresholds ($0.75$–$0.95$) is significantly more challenging than predicting enclosing rectangles.

---

### Section 4: Food Waste Estimation & Algorithmic Safeguards (Q23–Q26)

#### Q23: State the mathematical formula used for plate-area-based food waste estimation.
**Answer:**  
Following the research methodology of Ferreira et al. (MDPI Applied Sciences 2025):
$$\text{Food Waste Area Index (\%)} = \frac{\text{Area}_{\text{Food}}}{\text{Area}_{\text{Plate}} - \text{Area}_{\text{Garbage}}} \times 100$$
Where:
- $\text{Area}_{\text{Food}} = |\bigcup_{i \in \text{Edible}} M_i|$ (total pixel area covered by edible food items).
- $\text{Area}_{\text{Plate}} = |M_{\text{Plate}}|$ (total pixel area enclosed by the plate boundary).
- $\text{Area}_{\text{Garbage}} = |\bigcup_{j \in \text{Garbage}} M_j|$ (pixel area of inedible waste such as bones or toothpicks located on the plate).

#### Q24: What are the 9 algorithmic safeguards implemented in FOODWISE AI's waste calculator?
**Answer:**  
1. **Overlap Deduplication:** Uses boolean pixel raster union ($\bigcup M_i$) rather than summing raw pixel counts, preventing overlapping food masks from exceeding 100%.
2. **Minimum Plate Area Enforcement:** Enforces a minimum plate size threshold ($\ge 1000\text{ px}$) to prevent dividing by noise or false plate detections.
3. **Missing Plate Safeguard:** If no plate is detected, calculation defaults safely to $0.0\%$ with an explicit user notification rather than crashing.
4. **Non-Edible Utensil Exclusion:** Excludes cutlery (forks, knives, spoons), cups, and napkins from the food numerator.
5. **Garbage Subtraction:** Inedible food waste (bones, shells) is subtracted from the usable plate denominator.
6. **Division-by-Zero Guard:** Adds a numerical tolerance $\epsilon = 1\text{ px}$ to ensure the denominator is never zero.
7. **Negative Denominator Protection:** If garbage area occludes more than the plate area, clamps denominator to the plate area.
8. **Percentage Bounding:** Strictly clamps output within $[0.0\%, 100.0\%]$.
9. **Multi-Plate Spatial Filtering:** When multiple plates exist in an image, assigns food items to their enclosing plate polygon based on bounding-box intersection.

#### Q25: Why is it scientifically invalid to claim physical mass (grams) from a single 2D RGB photograph?
**Answer:**  
A standard 2D photograph provides only planar ($x, y$) spatial coordinates; it completely lacks the depth dimension ($z$-axis).  
Physical mass is governed by volume and physical density:
$$m = \rho \cdot V = \rho \int_{A} h(x, y) \, dA$$
Where $\rho$ is density ($\text{g/cm}^3$) and $h(x, y)$ is the height distribution of the food mound.  
A thin 1mm smear of tomato sauce and a 5cm tall mound of rice may exhibit identical 2D projected pixel surface areas on a plate, yet their physical masses differ by an order of magnitude. Without calibrated 3D depth sensors (RGB-D cameras, LiDAR) or reference volume markers, asserting physical grams from 2D images is scientifically unsupported.

#### Q26: How does the system handle Out-of-Distribution (OOD) images or empty plates?
**Answer:**  
1. **Empty / Clean Plate:** If the plate is detected with high confidence but no edible food masks satisfy the confidence threshold ($\tau \ge 0.25$), $\text{Area}_{\text{Food}} = 0$, yielding $\text{Waste (\%)} = 0.0\%$. The system displays a green "Clean Plate / Minimal Waste" badge.
2. **Out-of-Distribution Scene (e.g. desktop, room, random object):** If neither a plate nor any supported food item is detected, the pipeline catches the condition, rejects inference, and informs the user: *"No reliable canteen items detected. The image appears outside the model's supported distribution."*

---

### Section 5: Software Engineering, Database & SDG 12 Alignment (Q27–Q30)

#### Q27: How does the local SQLite database schema support cafeteria waste telemetry?
**Answer:**  
The local SQLite database (`data/foodwise.db`) stores structured records in a dedicated `waste_scans` table with the schema:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `timestamp` (DATETIME DEFAULT CURRENT_TIMESTAMP)
- `image_filename` (TEXT)
- `waste_percentage` (REAL)
- `food_area_pixels` (INTEGER)
- `plate_area_pixels` (INTEGER)
- `garbage_area_pixels` (INTEGER)
- `detected_items_json` (TEXT - JSON list of recognized items)
- `primary_category` (TEXT - predominant food group)
- `model_version` (TEXT - model weights used)
- `inference_time_ms` (REAL - latency tracking)

This enables longitudinal SQL queries, aggregation of clean-plate percentages, tracking of temporal trends across shifts, and auditing without requiring external network connectivity.

#### Q28: How does FOODWISE AI directly contribute to UN Sustainable Development Goal 12 (Target 12.3)?
**Answer:**  
**SDG Target 12.3** mandates halving global per capita food waste by 2030.  
FOODWISE AI contributes directly through:
1. **Granular Waste Telemetry:** Traditional waste tracking records gross bin weight, giving cafeteria managers no insight into which menu recipes diners rejected. FOODWISE AI pinpoints specific ingredients (e.g. 40% leftover boiled potatoes vs 2% leftover chicken).
2. **Portion Adjustment:** Enables dynamic portion resizing on frequently wasted side dishes.
3. **Procurement Optimization:** Prevents over-purchasing perishable items, mitigating methane emissions produced by decomposing organic waste in landfills.

#### Q29: What are the core components of the 5-page Streamlit application architecture?
**Answer:**  
1. **Home / Overview:** Interactive walkthrough of problem context, UN SDG 12 alignment, and YOLOv11 deep learning architecture.
2. **Plate Waste Detection:** Real-time upload interface supporting drag-and-drop or test dataset plates, dual visual outputs (annotated overlay and mask clustering), KPI cards, detection table, and database logging.
3. **Waste Analytics Dashboard:** Aggregated metrics, interactive Plotly timelines, food category breakdown charts, and CSV data export.
4. **Model Architecture & Evaluation:** In-depth technical breakdown of C3k2, SPPF, C2PSA, Proto heads, dataset audit charts, and formal evaluation metrics.
5. **Sustainability & SDG 12:** Scientific exploration of institutional cafeteria impact and volumetric depth considerations.

#### Q30: What engineering trade-offs were made between running YOLO11n on CPU versus training larger variants on GPU?
**Answer:**  
1. **Computational Feasibility:** YOLO11n-seg contains only 2.87M parameters and requires 9.8 GFLOPs, enabling local training on 12-core CPU in ~20 minutes with ~120ms inference latency per image.
2. **Deployment Portability:** Institutional cafeteria return stations often run on modest edge compute or POS hardware without discrete GPUs. YOLO11n-seg runs reliably on CPU-only edge devices without external cloud dependencies or network requirements.
3. **Benchmarking with High-Capacity Models:** For high-accuracy academic benchmarking, the system also incorporates the authors' pretrained YOLO11x-seg checkpoint (56.9M parameters, 194.9 GFLOPs), giving users the choice between lightweight edge speed and maximal precision.
