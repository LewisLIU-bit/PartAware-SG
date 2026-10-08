"""SAHI-inspired tiled masks and scale-aware metric sampling; no scene cues."""
import cv2
import numpy as np


def sampling_resolution(points):
    extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.
    return float(np.clip(extent/80., .002, .005)) if extent < .25 else .01


def association_radius(left, right):
    extent = min(float(np.max(np.ptp(left, axis=0))), float(np.max(np.ptp(right, axis=0))))
    return float(np.clip(.08*extent, .006, .02)) if extent < .25 else .1


def slice_boxes(height, width, size=512, overlap=.25):
    def starts(length):
        if length <= size:
            return [0]
        values = list(range(0, length-size+1, int(size*(1-overlap))))
        return sorted(set(values+[length-size]))
    return [(x, y, min(x+size, width), min(y+size, height))
            for y in starts(height) for x in starts(width)]


def proposals(model, image, names, device):
    height, width = image.shape[:2]
    regions = [(0, 0, width, height)]+slice_boxes(height, width)
    slice_count = len(regions)
    candidates = []
    for index, (x0, y0, x1, y1) in enumerate(regions):
        crop = image[y0:y1, x0:x1]
        result = model.predict(crop, conf=.22 if index == 0 else .3,
            iou=.6, imgsz=640, device=device, retina_masks=True, save=False, verbose=False)[0]
        if result.masks is None:
            continue
        if index == 0:
            for box in sorted(result.boxes, key=lambda value: -float(value.conf.item()))[:24]:
                bx0, by0, bx1, by1 = box.xyxy[0].cpu().numpy()
                bw, bh = bx1-bx0, by1-by0
                if bw*bh > .12*height*width or min(bw, bh) < 8:
                    continue
                margin = max(12, .25*max(bw, bh))
                region = (max(0, int(bx0-margin)), max(0, int(by0-margin)),
                          min(width, int(bx1+margin)), min(height, int(by1+margin)))
                if region not in regions:
                    regions.append(region)
        for raw, box in zip(result.masks.data.cpu().numpy(), result.boxes):
            mask = raw > .5
            if mask.shape != crop.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), crop.shape[1::-1],
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(mask.sum())
            if area < 32 or area > .8*mask.size:
                continue
            ys, xs = np.where(mask)
            if index and ((x0 and xs.min() < 3) or (y0 and ys.min() < 3)
                         or (x1 < width and xs.max() >= x1-x0-3)
                         or (y1 < height and ys.max() >= y1-y0-3)):
                continue
            if index and area > .2*height*width:
                continue
            full = np.zeros((height, width), bool)
            full[y0:y1, x0:x1] = mask
            bounds = box.xyxy[0].cpu().numpy()+[x0, y0, x0, y0]
            record = {'object_name': names[int(box.cls.item())],
                'confidence': float(box.conf.item()), 'segmentation_box': bounds.tolist(),
                'sam_quality_score': None,
                'proposal_sources': ['yoloe_zoom'] if index >= slice_count else (['yoloe_sahi'] if index else ['yoloe']),
                'confidence_type': 'uncalibrated_yoloe_score',
                'inference_crop': [x0, y0, x1, y1]}
            candidates.append((record, full))
    selected = []
    for record, mask in sorted(candidates, key=lambda value: -value[0]['confidence']):
        duplicate = False
        for _, existing in selected:
            inter = int(np.count_nonzero(mask & existing))
            union = int(mask.sum()+existing.sum()-inter)
            if inter/max(union, 1) >= .55:
                duplicate = True
                break
        if not duplicate:
            selected.append((record, mask))
    return selected
