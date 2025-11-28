from ultralytics import YOLO

class YOLORunner:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def predict(self, source, save=False):
        results = self.model(source, save=save)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            classes = r.boxes.cls.cpu().numpy()
            names = r.names  # dict of class idx to name
            detections.append({
                "classes": classes,
                "names": names
            })
        return detections
