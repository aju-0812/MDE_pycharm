#depthless object detection but voice tells depth --> video reference
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


# -------------------- YOLOv8 Wrapper --------------------
class YOLOv11:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections


# -------------------- MiDaS Wrapper --------------------
class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_HYBRID = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"


class Midas:
    def __init__(self, modelType=ModelType.DPT_HYBRID):
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if modelType in [ModelType.DPT_LARGE, ModelType.DPT_HYBRID]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

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
        depth_map_colored = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map_colored = cv2.applyColorMap(depth_map_colored, cv2.COLORMAP_INFERNO)
        real_depth = (known_height * focal_length) / (depth_map + 1e-6)

        return depth_map, depth_map_colored, real_depth

    def get_color_bar(self, height=480, width=60, min_depth=0, max_depth=10):
        color_bar = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
        color_bar = np.repeat(color_bar, width, axis=1)
        color_bar = cv2.applyColorMap(color_bar, cv2.COLORMAP_INFERNO)

        num_labels = 6
        for i, label in enumerate(np.linspace(min_depth, max_depth, num=num_labels)):
            y = int(height - (i / (num_labels - 1)) * height)
            text = f"{label:.1f}m"
            cv2.putText(color_bar, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return color_bar


# -------------------- Real-Time Vision + Speech --------------------
class RealTimeProcessing:
    def __init__(self, yolo_model="yolov8n.pt", depth_model=ModelType.DPT_HYBRID):
        self.yolo = YOLOv11(yolo_model)
        self.midas = Midas(depth_model)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.voice_queue = queue.Queue()
        self.speaker_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self.speaker_thread.start()

    def _speech_loop(self):
        while True:
            text, lang_code = self.voice_queue.get()
            asyncio.run(self._play_audio(text, lang_code))

    async def _play_audio(self, text, lang_code):
        voice_map = {
            "en": "en-US-AriaNeural",
            "ta": "ta-IN-PallaviNeural"
        }
        voice = voice_map.get(lang_code, "en-US-AriaNeural")
        file_name = f"{uuid.uuid4()}.mp3"
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(file_name)
            playsound(file_name)
        except Exception as e:
            print(f"[Error] Voice generation failed: {e}")
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)

    def speak_english(self, text):
        self.voice_queue.put((text, "en"))

    def speak_tamil(self, text):
        self.voice_queue.put((text, "ta"))

    def live_predict(self, video_source=0, output_video="output.mp4"):
        if isinstance(video_source, str) and not os.path.exists(video_source):
            print(f"[Error] Video file not found: {video_source}")
            return

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width * 2 + 30, frame_height))

        last_spoken_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.predict(frame)
            depth_map, depth_map_colored, depth_values = self.midas.predict(frame)

            annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections)

            if len(detections.xyxy) > 0:
                largest_idx = np.argmax([(box[2] - box[0]) * (box[3] - box[1]) for box in detections.xyxy])
                x1, y1, x2, y2 = map(int, detections.xyxy[largest_idx])
                object_depth = np.mean(depth_values[y1:y2, x1:x2])

                object_name = detections.class_name[largest_idx] if hasattr(detections, "class_name") else "object"

                tamil_dict = {
                    "person": "மனிதர்", "car": "கார்", "bicycle": "சைக்கிள்",
                    "bus": "பேருந்து", "truck": "லாரி", "motorcycle": "பைக்",
                    "dog": "நாய்", "cat": "பூனை"
                }
                object_name_ta = tamil_dict.get(object_name.lower(), "வாகனம்")

                en_msg = f"There is a {object_name} ahead. Depth is {object_depth:.2f} meters"
                ta_msg = f"முன்னாடி ஒரு {object_name_ta} இருக்கு. அதன் ஆழம் {object_depth:.2f} மீட்டர்"

                now = time.time()
                if now - last_spoken_time > 5:
                    self.speak_english(en_msg)
                    time.sleep(2)
                    self.speak_tamil(ta_msg)
                    last_spoken_time = now

            depth_bar = self.midas.get_color_bar(frame_height, 30, 0, 10)
            combined = np.hstack((annotated, depth_map_colored, depth_bar))

            out.write(combined)
            cv2.imshow("YOLO + MiDaS + Voice", combined)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


# -------------------- Run Application --------------------
if __name__ == "__main__":
    print("🔥 Launching Smart Vision System (YOLO + MiDaS + Edge-TTS)...")

    # Change the path to your video file here:
    input_video = "sample_input5.mp4"  # or use 0 for webcam

    processor = RealTimeProcessing(yolo_model="yolov8n.pt", depth_model=ModelType.MIDAS_SMALL)
    processor.live_predict(video_source=0, output_video="output.mp4")

