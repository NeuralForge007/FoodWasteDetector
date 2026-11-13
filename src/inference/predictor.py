"""
FOODWISE AI: Unified Predictor Pipeline
Processes input photographs through YOLOv11 instance segmentation, category mapping,
and plate-area-based food waste estimation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo_model import FoodWiseYOLOModel
from src.analysis.waste_calculator import WasteAreaCalculator
from src.config.category_mapping import (
    CLASS_ID_TO_NAME,
    CATEGORY_COLORS,
    get_category_for_class,
    get_item_semantic_type,
    is_edible_food
)

class FoodWisePredictor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        min_plate_area: int = 1000,
        device: str = "auto"
    ):
        self.yolo_model = FoodWiseYOLOModel(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            device=device
        )
        self.calculator = WasteAreaCalculator(min_plate_area=min_plate_area)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an input plate image.

        Returns:
            Dict containing:
                - 'annotated_image': PIL Image with masks, boxes, labels
                - 'clustering_image': PIL Image with masks only
                - 'detected_objects': List of object dicts
                - 'waste_analysis': Dict from WasteAreaCalculator
                - 'edible_items': List of edible food objects
                - 'table_rows': Formatted rows for UI display
                - 'has_reliable_detections': bool
                - 'message': str
        """
        # Convert input to RGB numpy array and PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                np_img = image_input.copy()
            pil_img = Image.fromarray(np_img)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h, w, _ = np_img.shape

        # Run YOLO inference
        results = self.yolo_model.predict(np_img, conf=conf_threshold)
        res = results[0]

        detected_objects = []
        table_rows = []

        if res.boxes is not None and len(res.boxes) > 0 and res.masks is not None:
            boxes = res.boxes
            masks = res.masks

            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = res.names.get(cls_id, CLASS_ID_TO_NAME.get(cls_id, f"item_{cls_id}"))

                # Mask extraction
                # masks.data[i] is at inference resolution; masks.xy contains polygon coordinates at original image resolution
                raw_mask = masks.data[i].cpu().numpy() # (H_inf, W_inf)
                resized_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                pixel_area = int(np.sum(resized_mask))

                category = get_category_for_class(cls_id)
                semantic_type = get_item_semantic_type(cls_id)
                edible = is_edible_food(cls_id)

                obj_info = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "mask": resized_mask,
                    "area": pixel_area,
                    "category": category,
                    "semantic_type": semantic_type,
                    "is_edible": edible
                }
                detected_objects.append(obj_info)

                table_rows.append({
                    "Food Item / Object": cls_name.title(),
                    "Category": category,
                    "Type": semantic_type,
                    "Confidence": f"{conf * 100:.1f}%",
                    "Pixel Area": f"{pixel_area:,} px"
                })

        # Calculate waste area index using research formula and safeguards
        waste_analysis = self.calculator.calculate_from_masks(
            detected_objects,
            image_shape=(h, w)
        )

        # Generate visualizations
        # 1. Annotated image using Ultralytics plot()
        annotated_array = res.plot(boxes=True, labels=True, masks=True)
        annotated_image = Image.fromarray(annotated_array)

        # 2. Clustering / mask-only image (research paper visualization)
        clustering_array = res.plot(boxes=False, labels=True, masks=True, color_mode="class")
        clustering_image = Image.fromarray(clustering_array)

        # Out-of-Domain and Low-Confidence Handling
        edible_items = [obj for obj in detected_objects if obj["is_edible"]]
        has_reliable_detections = len(detected_objects) > 0

        if not has_reliable_detections:
            message = "No reliable detection found for the uploaded image. The food may be outside the model's supported classes."
        elif not waste_analysis["plate_detected"]:
            message = waste_analysis["message"]
        else:
            message = f"Detected {len(edible_items)} edible food item(s) on plate. Estimated leftover food area: {waste_analysis['waste_percentage']:.1f}%."

        return {
            "annotated_image": annotated_image,
            "clustering_image": clustering_image,
            "detected_objects": detected_objects,
            "edible_items": edible_items,
            "waste_analysis": waste_analysis,
            "table_rows": table_rows,
            "has_reliable_detections": has_reliable_detections,
            "message": message,
            "image_dimensions": (w, h)
        }

