# Dataset Inspection and Annotation Audit Report

**Dataset Name:** PROJ3 – Food Waste Detection  
**Roboflow Workspace:** `project-eyfif`  
**Project ID:** `proj3-food-waste-detection`  
**Dataset Version:** Version 6  
**Format:** YOLOv8 / YOLOv11 Instance Segmentation (Polygons)  
**Licence:** CC BY 4.0 (Open Source Research Dataset)  

---

## 1. Summary Statistics

| Split | Number of Images | Percentage | Total Labels / Polygons | Purpose |
|---|---:|---:|---:|---|
| **Train** | 117 | 73.1% | Multiple instances | Model training and parameter updates |
| **Validation** | 35 | 21.9% | Multiple instances | Hyperparameter validation & early stopping |
| **Test** | 8 | 5.0% | Multiple instances | Final unbiased evaluation benchmark |
| **Total** | **160** | **100.0%** | **1150** | Full Roboflow v6 Dataset |

- **Image Dimensions:** {'640x640': 160}
- **Empty Label Files:** 0
- **Corrupted Images:** 0
- **Total Classes in Ontology:** 85

---

## 2. Verification of Rice and Core Food Classes

A major failure of previous attempts was the complete absence of `rice` in the segmentation annotations. In this verified Roboflow PROJ3 dataset (version 6):

| Class ID | Class Name | Category | Annotations | Images Containing Class | Status |
|---|---|---|---:|---:|---|
| **63** | `rice` | Cereals and Starches | **76** | **55** | Valid & Verified |
| **15** | `cabidela rice` | Cereals and Starches | **2** | **2** | Valid & Verified |
| **58** | `plate` | Non-edible Materials (Container) | **164** | **144** | Container Reference |
| **35** | `garbage` | Non-edible Materials (Waste) | **100** | **48** | Denominator Deduction |
| **20** | `chicken` | Protein and Meat | **23** | **16** | Food Item |
| **11** | `bread` | Cereals and Starches | **20** | **9** | Food Item |

Rice **does exist** with valid segmentation polygon annotations. When a user uploads an image containing rice, the model can now be trained to detect rice accurately.

---

## 3. Annotation Quality & Geometry Audit

1. **Polygon Structure:** Every annotation line follows the strict YOLO segmentation format `class_id x1 y1 x2 y2 ... xn yn`.
2. **Normalized Coordinates:** All vertex coordinates lie within $[0.0, 1.0]$.
3. **Polygon Fidelity:** Polygons closely follow the non-convex contours of food piles, plate rims, and cutlery.
4. **Visual Overlays:** 25 representative overlaid images have been generated and saved under `reports/annotation_visualizations/`.
5. **Contact Sheets:** A 30-image visual sheet has been generated at `reports/dataset_contact_sheet.png`.

---

## 4. Dataset Reconciliation with Published Literature

- **Base Canteen Images:** 175 base photographs were originally acquired in Portuguese university canteens.
- **Dataset Version 6 Composition:** Comprises 160 processed images (117 train, 35 valid, 8 test) featuring 85 verified classes.
- **Suitability Verdict:** The PROJ3 dataset version 6 contains real canteen tray images with authentic post-consumption plate leftovers, realistic lighting, and genuine plate-food configurations. It is **fully suitable** for reproducing and extending the research methodology.
