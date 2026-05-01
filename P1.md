## 🔹 P1: Smart Attendance & Attention Monitoring System

### 📌 Description

P1 is an advanced real-time system that not only detects faces but also analyzes **student engagement and behavior** using computer vision and deep learning.

It uses facial landmarks to determine whether a person is:

* **Attentive**
* **Distracted**
* **Inattentive**

---

### ⚙️ Core Functionalities

#### 🎥 Real-Time Face Detection

* Uses MTCNN for accurate face detection
* Optimized using frame scaling for performance

#### 🧠 Behavior Analysis

The system evaluates attention based on:

* 👁️ **Gaze Tracking**

  * Detects if the person is looking left, right, up, or down

* 🗣️ **Talking Detection**

  * Uses mouth movement variance to identify speaking

* 🏃 **Movement Tracking**

  * Detects sudden or continuous movement using centroid tracking

---

### 🧩 Multi-Person Tracking

* Assigns **unique IDs** to each detected individual
* Tracks them across frames using a custom **SmartTracker**
* Handles disappearance and reappearance of faces

---

### 📊 Attention Classification Logic

| Condition                  | Status      |
| -------------------------- | ----------- |
| Stable gaze + low movement | ATTENTIVE   |
| Looking away / talking     | DISTRACTED  |
| Excessive movement         | INATTENTIVE |

---

### 📈 Session Analytics

* Tracks:

  * Focused time
  * Distracted time
* Generates a **CSV report** automatically:

```
Attendance_Report_<timestamp>.csv
```

#### Report Includes:

* Student ID
* Focused Time
* Distracted Time
* Engagement Score (%)
* Final Status

---

### ⚡ Performance Optimizations

* Asynchronous face detection (threading)
* Frame resizing for faster processing
* Buffered status smoothing (deque + majority voting)

---

### 🖥️ Output Features

* Bounding boxes around faces
* Real-time labels (ATTENTIVE / DISTRACTED / INATTENTIVE)
* FPS counter
* Student count
* Visual gaze lines (debug mode)

---

### ⌨️ Controls

* Press **'q'** → Exit application

---

### 🔧 Technologies Used

* Python
* OpenCV
* MTCNN
* TensorFlow
* NumPy

---

### 💡 Key Highlights (Important for Viva)

* Custom tracking algorithm (not basic OpenCV tracker)
* Behavioral analytics using facial keypoints
* Real-time performance optimization using threading
* Automatic report generation system

---

### 🚀 Possible Improvements

* Face Recognition (identify students by name)
* GUI Dashboard for analytics
* Database integration (store attendance permanently)
* Emotion detection (happy, sad, etc.)
* Web-based interface

---
