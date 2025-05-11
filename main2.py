# scale and depth valued map    -----> single depth video reference
import cv2
import torch
import numpy as np
import supervision as sv
from enum import Enum
from ultralytics import YOLO
import os


# YOLO Model Setup
class YOLOv11:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, frame):
        results = self.model(frame)[0]  # Run inference on frame
        detections = sv.Detections.from_ultralytics(results)  # Convert results to supervision format
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
        if modelType.value in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, frame, focal_length=700, known_height=0.25):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),  # [1, 1, H, W]
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        np.save("depth_map8 .npy", depth_map)

        # Normalize depth map for visualization
        depth_map_colored = cv2.normalize(depth_map,None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map_colored = cv2.applyColorMap(depth_map_colored,
                                              cv2.COLORMAP_INFERNO)  # Inferno (black → red → yellow → white)

        # Convert depth map to real-world distance
        real_depth = (known_height * focal_length) / (depth_map + 1e-6)  # Avoid division by zero

        return depth_map, depth_map_colored, real_depth

    def get_color_bar(self, height=480, width=60, min_depth=0, max_depth=10):
        # Create a vertical gradient bar
        color_bar = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
        color_bar = np.repeat(color_bar, width, axis=1)
        color_bar = cv2.applyColorMap(color_bar, cv2.COLORMAP_INFERNO)

        mask = np.zeros_like(color_bar, dtype=np.uint8)
        mask = cv2.rectangle(mask, (0, 10), (width, height - 10), (255, 255, 255), thickness=-1)
        color_bar = cv2.bitwise_and(color_bar, mask)

        # Add Depth Labels
        num_labels = 6
        for i, label in enumerate(np.linspace(min_depth, max_depth, num=num_labels)):
            y = int(height - (i / (num_labels - 1)) * height)  # Scale positions
            text = f"{label:.1f}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = (width - text_size[0]) // 2

            cv2.putText(color_bar, text, (text_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return color_bar

    def add_depth_values(self, frame, depth_map, depth_values, scale_factor=10):
        for i in range(0, depth_map.shape[0], scale_factor):
            for j in range(0, depth_map.shape[1], scale_factor):
                depth = depth_values[i, j]
                cv2.putText(frame, f"{depth:.2f}", (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


# Real-Time Processing
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

        cap = cv2.VideoCapture(video_source)  # Use 0 for webcam
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width * 2 + 30, frame_height))  # Extra space for scale

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.predict(frame)

            # MiDaS Depth Estimation
            depth_map, depth_map_colored, depth_values = self.midas.predict(frame)

            # Annotate detections with bounding boxes and labels
            annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Side Depth Scale
            depth_color_bar = self.midas.get_color_bar(height=frame_height, width=50, min_depth=np.min(depth_values),
                                                       max_depth=np.max(depth_values))

            # Focus on Main Object
            if len(detections.xyxy) > 0:
                # Sort detections by size (biggest object first)
                largest_obj = max(detections.xyxy, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

                x1, y1, x2, y2 = map(int, largest_obj)  # Bounding box
                object_depth = np.mean(depth_values[y1:y2, x1:x2])  # Average depth of object

                # Draw Depth Value on Object
                cv2.rectangle(annotated_frame, (x1, y1 - 40), (x1 + 150, y1 - 10), (0, 0, 0),
                              -1)  # Black box for clarity
                cv2.putText(annotated_frame, f"Depth: {object_depth:.2f}m", (x1 + 5, y1 - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Stack images side-by-side (YOLO output, Depth Map, Depth Scale)
            depth_color_bar = self.midas.get_color_bar(height=frame_height, width=30, min_depth=0, max_depth=10)
            combined = np.hstack((annotated_frame, depth_map_colored, depth_color_bar))

            # Save frame to video
            out.write(combined)

            # Display the frame
            cv2.imshow("YOLOv11 + MiDaS Depth Estimation with Scale & Focus", combined)

            # Exit condition
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break



        cap.release()
        out.release()
        cv2.destroyAllWindows()


# Run Application
if __name__ == "__main__":
    print("🔥 Launching Smart Vision System (YOLO + MiDaS + Edge-TTS)...")

    # Change the path to your video file here:
    input_video = "sample_input7.mp4"  # or use 0 for webcam

    processor = RealTimeProcessing(yolo_model="yolov8n.pt", depth_model=ModelType.DPT_LARGE)
    processor.live_predict(video_source=input_video, output_video="output.mp4")