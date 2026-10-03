"""GroundingDINO category inference and SAM box segmentation.

SAM quality ranks masks for a supplied box; it is not a category confidence.
The old reviewed-candidate and long-description experiment methods are removed.
"""
import os
import sys
import json

_here = os.path.dirname(__file__)
_candidates = [
    os.path.abspath(os.path.join(_here, "../../thirdparty/Grounded-Segment-Anything")),
    os.path.abspath(os.path.join(_here, "../../../thirdparty/Grounded-Segment-Anything")),
]
path = next((p for p in _candidates if os.path.isdir(p)), _candidates[0])
print(path)

# Ensure we import the vendored repos (avoid pip name conflicts).
sys.path.insert(0, os.path.join(path, "segment_anything"))
sys.path.insert(0, os.path.join(path, "GroundingDINO"))
sys.path.insert(0, path)
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from io import BytesIO


def save_detection_stage(
    image,
    classes,
    detections,
    source_ids,
    debug_dir,
    stage,
    thresholds,
):
    """Save detection records and labeled boxes without changing results."""
    if debug_dir is None:
        return

    os.makedirs(debug_dir, exist_ok=True)
    canvas = image.copy()
    height, width = image.shape[:2]
    records = []

    for index, box in enumerate(detections.xyxy):
        raw_class_id = detections.class_id[index]
        class_id = (
            int(raw_class_id) if raw_class_id is not None else None
        )

        if class_id is not None and 0 <= class_id < len(classes):
            name = classes[class_id]
        else:
            name = "<unmatched>"

        score = float(detections.confidence[index])
        source_id = int(source_ids[index])

        record = {
            "source_id": source_id,
            "class_id": class_id,
            "name": name,
            "confidence": score,
            "xyxy": [float(value) for value in box],
        }

        if detections.mask is not None:
            record["mask_pixels"] = int(
                np.count_nonzero(detections.mask[index])
            )

        records.append(record)

        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), (0, 255, 255), 2
        )

        # Keep the label visible when the box touches an image boundary.
        label = f"#{source_id} {name} {score:.4f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        text_x = max(0, min(x1, width - text_width - 4))
        text_y = max(
            text_height + 4,
            min(y1 - 5, height - baseline - 2),
        )

        cv2.rectangle(
            canvas,
            (text_x, text_y - text_height - 3),
            (text_x + text_width + 3, text_y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (text_x + 1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    payload = {
        "stage": stage,
        "classes": list(classes),
        "image_size": {"width": width, "height": height},
        "thresholds": thresholds,
        "detections": records,
    }

    json_path = os.path.join(debug_dir, f"{stage}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    image_path = os.path.join(debug_dir, f"{stage}.jpg")
    if not cv2.imwrite(image_path, canvas):
        raise RuntimeError(f"Failed to save diagnostic image: {image_path}")
class GroundedSam:
    def __init__(self, load_dino=True) -> None:
        # Device
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = path + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = path + "/groundingdino_swint_ogc.pth"
        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = path + "/sam_vit_h_4b8939.pth"
        # Building GroundingDINO inference model
        self.grounding_dino_model = None
        if load_dino:
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            )
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

    @torch.inference_mode()
    def segment_external_boxes(self, image, objects, outside_penalty=0.5):
        """Segment source-pixel boxes without detector scores or features."""
        if not np.isfinite(outside_penalty) or outside_penalty < 0:
            raise ValueError("outside_penalty must be finite and nonnegative")

        height, width = image.shape[:2]
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")

        boxes, seen = [], set()
        for obj in objects:
            qid = obj.get("query_id")
            if not isinstance(qid, str) or not qid.strip() or qid in seen:
                raise ValueError(f"Missing or duplicate query_id: {qid}")
            seen.add(qid)

            if not isinstance(obj.get("name"), str) or not obj["name"].strip():
                raise ValueError(f"Missing name: {qid}")

            box = np.asarray(obj.get("bbox_xyxy"), dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError(f"Invalid box: {qid}")

            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Box outside source image: {qid}: {box}")
            boxes.append(box)

        selected = np.zeros((len(objects), height, width), dtype=bool)
        details = []

        if objects:
            self.sam_predictor.set_image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

        for i, (obj, box) in enumerate(zip(objects, boxes)):
            masks, quality, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )
            masks = np.asarray(masks, dtype=bool)
            quality = np.asarray(quality).reshape(-1)

            if masks.shape != (len(quality), height, width):
                raise RuntimeError(f"Invalid SAM output: {obj['query_id']}")

            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)

            options = []
            for k, mask in enumerate(masks):
                area = int(mask.sum())
                if area == 0 or not np.isfinite(quality[k]):
                    continue

                inside = int(mask[y1:y2, x1:x2].sum())
                leak = 1.0 - inside / area

                options.append({
                    "candidate_index": k,
                    "sam_quality_score": float(quality[k]),
                    "outside_box_fraction": leak,
                    "selection_score": (
                        float(quality[k]) - outside_penalty * leak
                    ),
                    "mask_pixels": area,
                })

            meta = {
                "proposal": dict(obj),
                "mask_index": i,
                "sam_prompt": "box_only",
                "sam_options": options,
                "selected_candidate_index": None,
                "status": "no_valid_mask",
            }

            if options:
                best = max(options, key=lambda item: item["selection_score"])
                selected[i] = masks[best["candidate_index"]]
                meta.update(best)
                meta["selected_candidate_index"] = best["candidate_index"]
                meta["status"] = "segmented_unverified"

            details.append(meta)

        return selected, details

    def generate_grid_points(self, box, num_points=9):
        """
        Generate a grid of points within the box, centered around the middle.
        num_points should be a perfect square (4, 9, 16, etc.)
        """
        x1, y1, x2, y2 = box
        grid_size = int(np.sqrt(num_points))
        
        # Calculate step sizes
        width = x2 - x1
        height = y2 - y1
        x_step = width / (grid_size + 1)
        y_step = height / (grid_size + 1)
        
        points = []
        # Generate grid points
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = x1 + j * x_step
                y = y1 + i * y_step
                points.append([x, y])
        
        return np.array(points)

    # Prompting SAM with detected boxes
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []

        for box in xyxy:
            # ADDED point prompt by CHG
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            point_coords = np.array([[x_center, y_center]])
            point_labels = np.array([1])
            

            masks, scores, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    # Example: img_io contains the image as a BytesIO object
    # You can generate it using PIL
    def bytesio_to_cv2(self, img_io):
        """
        Convert an image in BytesIO to an OpenCV image.
        :param img_io: BytesIO object containing the image.
        :return: OpenCV image (numpy array).
        """
        # Read the bytes from BytesIO
        img_bytes = img_io.getvalue()
        
        # Convert the bytes to a numpy array
        np_array = np.frombuffer(img_bytes, np.uint8)
        
        # Decode the numpy array into an OpenCV image
        cv2_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_COLOR for color images
        return cv2_image
    
    def infer(self, image, classes, box_threshold, text_threshold, nms_threshold, confidence_threshold=0.5, image_type="cv2",debug_dir=None):
        # load image
        if image_type == "cv2":
            pass
        elif image_type == "bytesio":        
            image = self.bytesio_to_cv2(image)
        else:
            raise ValueError(f"Invalid image type: {image_type}")
        # cv2.imshow("image", image)  # DO NOT USE THIS!!! Otherwise the following GPU task will be blocked
        # cv2.waitKey(0)
        # detect objects
        detections, features = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        source_ids = np.arange(len(detections.xyxy), dtype=np.int64)

        thresholds = {
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "confidence_threshold": float(confidence_threshold),
            "nms_threshold": float(nms_threshold),
        }

        save_detection_stage(
            image, classes, detections, source_ids,
            debug_dir, "01_dino_candidates", thresholds,
        )
        # print(f"Detections: {detections}")
        # print(f"Features: {features}")

        # filter out detections with low confidence
        filter_ids = [i for i, confidence in enumerate(detections.confidence) if confidence > confidence_threshold]
        detections = detections[np.array(filter_ids, dtype=int)]
        features = features[filter_ids]

        source_ids = source_ids[np.asarray(filter_ids, dtype=int)]

        save_detection_stage(
            image, classes, detections, source_ids,
            debug_dir, "02_after_confidence", thresholds,
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[int(class_id)]} {float(confidence):0.2f}"
            for confidence, class_id
            in zip(detections.confidence, detections.class_id)
        ]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        try:
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        except Exception:
            # Older/newer supervision variants may not have LabelAnnotator or may differ in signature.
            pass
        # save the annotated grounding dino image
        # cv2.imwrite(path + "/" + "groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post process
        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # print(f"After NMS: {len(detections.xyxy)} boxes")
        features = features[nms_idx] # only keep the features of the NMSed detections
        source_ids = source_ids[np.asarray(nms_idx, dtype=int)]

        save_detection_stage(
            image, classes, detections, source_ids,
            debug_dir, "03_after_nms", thresholds,
        )

        # convert detections to masks
        if len(detections.xyxy) > 0:
            detections.mask = self.segment(
                sam_predictor=self.sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )
        else:
            # Preserve a valid mask shape for frames with no detections.
            detections.mask = np.zeros(
                (0, image.shape[0], image.shape[1]),
                dtype=bool,
            )

        save_detection_stage(
            image, classes, detections, source_ids,
            debug_dir, "04_after_sam", thresholds,
        )

        if debug_dir is not None:
            np.savez_compressed(
                os.path.join(debug_dir, "04_sam_masks.npz"),
                masks=detections.mask.astype(bool),
                source_ids=source_ids,
                boxes=detections.xyxy,
                confidences=detections.confidence,
                class_ids=np.asarray(
                    [
                        -1 if value is None else int(value)
                        for value in detections.class_id
                    ],
                    dtype=np.int64,
                ),
            )
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{classes[int(class_id)]} {float(confidence):0.2f}" 
            for confidence, class_id 
            in zip(detections.confidence, detections.class_id)]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        try:
            label_annotator = sv.LabelAnnotator()
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        except Exception:
            pass

        # save the annotated grounded-sam image
        # cv2.imwrite(path + "/" + "grounded_sam_annotated_image.jpg", annotated_image)

        if image_type == "bytesio":
            # Turn BGR to RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image, detections.mask, detections.class_id, detections.confidence, features.cpu().numpy()
