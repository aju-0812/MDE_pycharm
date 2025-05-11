# YOLO + MiDaS + DeepSORT + EdgeTTS Real-Time Processor

import cv2
import torch
import numpy as np
import supervision as sv
import asyncio
import edge_tts
from playsound import playsound
import uuid
import os
import time
import threading
from enum import Enum
from ultralytics import YOLO
import queue
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- YOLOv8 Wrapper --------------------
class YOLOv8:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def __call__(self, image):
        return self.model(image)[0]

# -------------------- MiDaS Depth Estimator --------------------
class MiDaS:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if self.model_type in ["DPT_Large", "DPT_Hybrid"] else torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def estimate_depth(self, image):
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
            return prediction.cpu().numpy()

    def to_colormap(self, depth_map):
        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

    def get_color_bar(self, height=480):
        gradient = np.linspace(1, 0, height).reshape(-1, 1)
        color_bar = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return color_bar

# -------------------- EdgeTTS Voice Alert --------------------
class Voice(Enum):
    ENGLISH = "en-US-AriaNeural"
    TAMIL = "ta-IN-PallaviNeural"

class EdgeTTS:
    def __init__(self, voice=Voice.ENGLISH):
        self.voice = voice
        self.queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    async def _speak(self, text):
        filename = f"{uuid.uuid4()}.mp3"
        try:
            communicate = edge_tts.Communicate(text, self.voice.value)
            await communicate.save(filename)
            playsound(filename)
        except Exception as e:
            print("TTS Error:", e)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def speak(self, text):
        self.queue.put(text)

    def _run(self):
        while True:
            text = self.queue.get()
            asyncio.run(self._speak(text))

# -------------------- Combined Processor --------------------
class RealTimeProcessor:
    def __init__(self, video_path, yolo_model="yolov8n.pt"):
        self.cap = cv2.VideoCapture(video_path)
        self.yolo = YOLOv8(yolo_model)
        self.midas = MiDaS()
        self.tracker = DeepSort(max_age=30)
        self.box_annotator = sv.BoxAnnotator()
        self.tts = EdgeTTS(Voice.ENGLISH)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter('output_with_depth_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2 + 50, frame_height))

    def run(self):
        if not self.cap.isOpened():
            print("❌ Error: Could not open video. Check the path.")
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠️ Could not read frame.")
                break

            print("✅ Processing frame")

            detections = self.yolo(frame)
            depth_map = self.midas.estimate_depth(frame)
            depth_colored = self.midas.to_colormap(depth_map)

            if detections.boxes is not None:
                confs = detections.boxes.conf
                mask = confs > 0.3
                detections_filtered = detections.boxes[mask]
            else:
                detections_filtered = None

            if detections_filtered is not None and len(detections_filtered) > 0:
                # Ensure detections are in the correct format
                xyxy = detections_filtered.xyxy.cpu().numpy()  # Shape should be (n, 4) for n boxes
                confidences = detections_filtered.conf.cpu().numpy()  # Confidence values
                class_ids = detections_filtered.cls.cpu().numpy()  # Class IDs for each detection

                # Convert xyxy to a 2D list format
                xyxy = xyxy.tolist()  # Convert the numpy array to a list of lists

                # Debugging print to ensure format is correct
                print("xyxy (list of lists):", xyxy)
                print("confidences:", confidences)
                print("class_ids:", class_ids)
            else:
                xyxy = []
                confidences = []
                class_ids = []

            # Ensure that the xyxy is correctly formatted as a list of lists of 4 values
            for detection in xyxy:
                if isinstance(detection, list) and len(detection) == 4:
                    print(f"Valid detection: {detection}")
                else:
                    print(f"Invalid detection format: {detection}")

            # Pass the data to the tracker
            tracks = self.tracker.update_tracks(xyxy, confidences, class_ids)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                try:
                    depth_value = depth_map[center_y, center_x]
                except IndexError:
                    print(f"⚠️ Skipping out-of-bounds depth value for track {track_id}")
                    continue

                label = f"ID:{int(track_id)} | {depth_value:.2f}m"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            combined = np.hstack((frame, depth_colored, self.midas.get_color_bar(frame.shape[0])))
            self.out.write(combined)
            cv2.imshow("YOLO + MiDaS + DeepSORT", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = RealTimeProcessor("sample_input2.mp4", "yolov8n.pt")
    processor.run()
