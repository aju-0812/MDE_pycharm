# multi depth with video inference
import cv2
import torch
import numpy as np
import supervision as sv
from enum import Enum
from ultralytics import YOLO
import os
import timm.layers

# YOLO Wrapper
class YOLOv11:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.4):
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.conf_threshold = confidence_threshold

    def predict(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter based on confidence
        detections = detections[detections.confidence > self.conf_threshold]
        return detections

# MiDaS Depth Estimation
class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_HYBRID = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas:
    def __init__(self, modelType=ModelType.DPT_LARGE):
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device).eval()

        # Load MiDaS Transformations
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (
            midas_transforms.dpt_transform if modelType.value in ["DPT_Large", "DPT_Hybrid"]
            else midas_transforms.small_transform
        )

    def predict(self, frame, focal_length=700, known_height=0.25):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        np.save("multidepth8.npy", depth_map)

        # Apply median blur for noise reduction
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)

        # Convert depth to distance (arbitrary scale)
        real_depth = (known_height * focal_length) / (depth_map + 1e-6)

        # Color map for visualization
        depth_colored = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_INFERNO)

        return depth_map, depth_colored, real_depth

    def get_color_bar(self, height=480, width=60, min_depth=0, max_depth=10):
        bar = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
        bar = np.repeat(bar, width, axis=1)
        bar = cv2.applyColorMap(bar, cv2.COLORMAP_INFERNO)
        for i, val in enumerate(np.linspace(min_depth, max_depth, 6)):
            y = int(height - (i / 5) * height)
            label = f"{val:.1f}m"
            cv2.putText(bar, label, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return bar

# Main Processing Class
class RealTimeProcessing:
    def __init__(self, yolo_model="yolov8n.pt", depth_model=ModelType.DPT_LARGE):
        self.yolo = YOLOv11(yolo_model)
        self.midas = Midas(depth_model)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def live_predict(self, video_source=0, output_video="output.avi"):
        if isinstance(video_source, str) and not os.path.exists(video_source):
            print(f"[Error] Video file not found: {video_source}")
            return

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (frame_width * 2 + 30, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.predict(frame)
            depth_map, depth_colored, depth_values = self.midas.predict(frame)

            annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Draw depth per object
            for xyxy in detections.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                roi_depth = depth_values[y1:y2, x1:x2]
                object_depth = np.mean(roi_depth[roi_depth > 0])  # Avoid 0-depth

                # Draw Depth Box
                cv2.rectangle(annotated_frame, (x1, y1 - 40), (x1 + 160, y1 - 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Depth: {object_depth:.2f}m", (x1 + 5, y1 - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            # Append scale bar
            depth_bar = self.midas.get_color_bar(height=frame_height, width=30,
                                                 min_depth=np.min(depth_values),
                                                 max_depth=np.max(depth_values))

            combined = np.hstack((annotated_frame, depth_colored, depth_bar))
            out.write(combined)
            cv2.imshow("YOLO + MiDaS: Object Depth Estimation", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Run
if __name__ == "__main__":
    print("🚀 Starting Smart Depth-Aware Object Detection System...")
    input_video = "sample_input10.mp4"  # or use 0 for webcam
    processor = RealTimeProcessing(yolo_model="yolov8n.pt", depth_model=ModelType.DPT_LARGE)
    processor.live_predict(video_source=input_video, output_video="output_depth.mp4")
