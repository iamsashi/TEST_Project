import cv2
import math
import os
import time
import csv
import threading
import numpy as np
from collections import deque, Counter

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    from mtcnn import MTCNN
except ImportError:
    print("❌ ERROR: MTCNN not installed. Run 'pip install mtcnn tensorflow'")
    exit()

# ================= 1. CALIBRATED CONFIGURATION =================
MOVEMENT_THRESHOLD = 4        
INSTANT_MOVE_LIMIT = 25       
TALKING_THRESHOLD = 20.0      

# Gaze Settings
GAZE_LEFT_LIMIT = 1.35   
GAZE_RIGHT_LIMIT = 0.65
GAZE_UP_LIMIT = 0.35
GAZE_DOWN_LIMIT = 1.65

HISTORY_LEN = 15              
CONFIDENCE_THRESHOLD = 0.95   
PROCESS_SCALE = 0.5           

# ================= 2. ASYNC DETECTOR =================
class AsyncFaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.frame = None
        self.detections = []
        self.stopped = False
        self.lock = threading.Lock()
        self.new_data_available = False

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def set_frame(self, frame):
        if not self.new_data_available:
            with self.lock:
                self.frame = frame.copy() 
                self.new_data_available = True

    def update(self):
        while not self.stopped:
            if self.new_data_available:
                with self.lock:
                    input_frame = self.frame
                    self.new_data_available = False
                
                h, w = input_frame.shape[:2]
                small_frame = cv2.resize(input_frame, (int(w * PROCESS_SCALE), int(h * PROCESS_SCALE)))
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                try:
                    results = self.detector.detect_faces(rgb_small)
                except Exception:
                    results = []

                scaled_results = []
                scale = 1 / PROCESS_SCALE
                for face in results:
                    x, y, w, h = face['box']
                    face['box'] = [int(x*scale), int(y*scale), int(w*scale), int(h*scale)]
                    kps = face['keypoints']
                    for key in kps:
                        kps[key] = (int(kps[key][0]*scale), int(kps[key][1]*scale))
                    face['keypoints'] = kps
                    if face['confidence'] >= CONFIDENCE_THRESHOLD:
                        scaled_results.append(face)

                with self.lock:
                    self.detections = scaled_results

    def get_detections(self):
        with self.lock:
            return self.detections

    def stop(self):
        self.stopped = True

# ================= 3. ANALYTICS MANAGER =================
class SessionManager:
    def __init__(self):
        self.stats = {} 

    def update_stats(self, obj_id, status):
        if obj_id not in self.stats:
            self.stats[obj_id] = {'focused': 0, 'distracted': 0}
        
        if status == "ATTENTIVE":
            self.stats[obj_id]['focused'] += 0.1
        else:
            self.stats[obj_id]['distracted'] += 0.1

    def save_report(self):
        filename = f"Attendance_Report_{int(time.time())}.csv"
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Student ID", "Focused Time (s)", "Distracted Time (s)", "Engagement Score (%)", "Status"])
                
                for s_id, data in self.stats.items():
                    focused = round(data['focused'], 2)
                    distracted = round(data['distracted'], 2)
                    total = focused + distracted
                    score = int((focused / total) * 100) if total > 0 else 0
                    final_status = "Attentive" if score > 75 else "Inattentive"
                    writer.writerow([f"ID {s_id}", focused, distracted, f"{score}%", final_status])
            print(f"\n✅ Report Saved: {filename}")
        except Exception:
            pass

# ================= 4. STABILIZED TRACKER (FIXED) =================
class SmartTracker:
    def __init__(self, session_manager):
        self.next_id = 1
        self.objects = {}           
        self.disappeared = {}       
        self.movement_scores = {}
        self.mouth_history = {} 
        self.status_buffer = {} 
        self.session = session_manager
        self.debug_info = {} 

    def update(self, detections):
        input_centroids = []
        input_keypoints = []
        
        for face in detections:
            x, y, w, h = face['box']
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            input_centroids.append((cX, cY))
            input_keypoints.append(face['keypoints'])

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        matches = []
        
        # --- FIX 1: UNPACK 3 VALUES (x, y, status) ---
        for i, (ox, oy, _) in enumerate(object_centroids):
            for j, (nx, ny) in enumerate(input_centroids):
                dist = math.hypot(ox - nx, oy - ny)
                matches.append((object_ids[i], j, dist))
        matches.sort(key=lambda x: x[2])

        used_old = set()
        used_new = set()

        for obj_id, idx, dist in matches:
            if obj_id in used_old or idx in used_new: continue
            if dist > 300: continue

            # --- FIX 2: UNPACK 3 VALUES ---
            old_x, old_y, _ = self.objects[obj_id]
            new_x, new_y = input_centroids[idx]
            self.disappeared[obj_id] = 0

            # --- FEATURE ANALYSIS ---
            kps = input_keypoints[idx]
            nose = kps['nose']
            mouth_l = kps['mouth_left']
            mouth_r = kps['mouth_right']
            left_eye = kps['left_eye']
            right_eye = kps['right_eye']
            
            # Talking Check
            mouth_mid_y = (mouth_l[1] + mouth_r[1]) / 2
            jaw_dist = abs(nose[1] - mouth_mid_y)
            self.mouth_history[obj_id].append(jaw_dist)
            
            # Gaze Check
            dist_l_n = math.hypot(left_eye[0] - nose[0], left_eye[1] - nose[1])
            dist_r_n = math.hypot(right_eye[0] - nose[0], right_eye[1] - nose[1])
            try: h_ratio = dist_l_n / dist_r_n
            except: h_ratio = 1.0

            eye_mid_y = (left_eye[1] + right_eye[1]) / 2
            eye_to_nose = abs(nose[1] - eye_mid_y)
            try: v_ratio = eye_to_nose / jaw_dist
            except: v_ratio = 1.0

            # Movement Check
            raw_dist = math.hypot(new_x - old_x, new_y - old_y)
            score = self.movement_scores.get(obj_id, 0)
            if raw_dist > INSTANT_MOVE_LIMIT: score += 50
            elif raw_dist > MOVEMENT_THRESHOLD: score += raw_dist
            else: score = max(0, score - 5)
            self.movement_scores[obj_id] = min(score, 300)

            # --- DETERMINE STATUS ---
            current_status = "ATTENTIVE" 
            
            # talking variance
            if np.var(list(self.mouth_history[obj_id])) > TALKING_THRESHOLD:
                current_status = "DISTRACTED" 
            elif h_ratio < GAZE_RIGHT_LIMIT or h_ratio > GAZE_LEFT_LIMIT:
                current_status = "DISTRACTED" 
            elif v_ratio < GAZE_UP_LIMIT or v_ratio > GAZE_DOWN_LIMIT:
                current_status = "DISTRACTED" 
            elif score > 100:
                current_status = "INATTENTIVE" 

            self.status_buffer[obj_id].append(current_status)
            stable_status = Counter(self.status_buffer[obj_id]).most_common(1)[0][0]
            
            # SAVE (x, y, status)
            self.objects[obj_id] = (new_x, new_y, stable_status)

            self.debug_info[obj_id] = {'nose': nose, 'l_eye': left_eye, 'r_eye': right_eye}
            used_old.add(obj_id)
            used_new.add(idx)

        for obj_id in list(self.objects.keys()):
            if obj_id not in used_old:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 50: self.deregister(obj_id)

        for i in range(len(input_centroids)):
            if i not in used_new: self.register(input_centroids[i])
            
        return self.objects

    def register(self, centroid):
        self.objects[self.next_id] = (centroid[0], centroid[1], "ATTENTIVE")
        self.disappeared[self.next_id] = 0
        self.movement_scores[self.next_id] = 0
        self.mouth_history[self.next_id] = deque(maxlen=20) 
        self.status_buffer[self.next_id] = deque(maxlen=HISTORY_LEN) 
        for _ in range(HISTORY_LEN): self.status_buffer[self.next_id].append("ATTENTIVE")
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.movement_scores[obj_id]
        del self.mouth_history[obj_id]
        del self.status_buffer[obj_id]
        if obj_id in self.debug_info: del self.debug_info[obj_id]

# ================= 5. MAIN SYSTEM =================
if __name__ == "__main__":
    print("Initializing System...")
    async_detector = AsyncFaceDetector().start()
    session = SessionManager()
    tracker = SmartTracker(session)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    prev_time = 0

    print("✅ System Ready.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            async_detector.set_frame(frame)
            detections = async_detector.get_detections()
            tracker.update(detections)
            
            for face in detections:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2) 

            for obj_id, data in tracker.objects.items():
                cx, cy, status = data 
                
                session.update_stats(obj_id, status)
                
                if status == "DISTRACTED":
                    color = (0, 0, 255) 
                    label = "DISTRACTED"
                elif status == "INATTENTIVE":
                    color = (0, 140, 255) 
                    label = "INATTENTIVE"
                else:
                    color = (0, 255, 0) 
                    label = "ATTENTIVE"

                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"ID {obj_id}", (cx, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, label, (cx - 40, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if obj_id in tracker.debug_info:
                    d = tracker.debug_info[obj_id]
                    cv2.line(frame, d['l_eye'], d['nose'], (0, 255, 255), 1)
                    cv2.line(frame, d['r_eye'], d['nose'], (0, 255, 255), 1)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.rectangle(frame, (0,0), (200, 40), (0,0,0), -1)
            cv2.putText(frame, f"Students: {len(tracker.objects)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Smart Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        async_detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        session.save_report()