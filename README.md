# 🚦 AI Based Traffic Signal Control System 

An **AI-powered Smart Traffic Management System** that detects vehicles in real-time using **YOLOv8**, analyzes traffic density, dynamically controls traffic signals, stores historical data, and predicts future traffic patterns using machine learning.

This project is built with **Python, Flask, OpenCV, YOLOv8, and SQLite** and provides a **web-based dashboard** to monitor traffic conditions.

---

# 📌 Features

### 🚗 Real-Time Vehicle Detection

* Uses **YOLOv8 object detection** to identify vehicles:

  * Cars
  * Motorcycles
  * Buses
  * Trucks
* Works with:

  * Live webcam
  * Uploaded traffic videos

---

### 🚦 Smart Traffic Signal Control

Traffic lights automatically adjust based on traffic density.

| Density | RED    | GREEN  |
| ------- | ------ | ------ |
| LOW     | 30 sec | 10 sec |
| MEDIUM  | 15 sec | 15 sec |
| HIGH    | 10 sec | 30 sec |

Includes:

* **AUTO mode** (AI controlled)
* **MANUAL mode** (user controlled)

---

### 📊 Traffic Analytics

The system records traffic data in **SQLite database** and generates:

* Vehicle statistics
* Hourly traffic patterns
* Density distribution
* Peak traffic hours
* Historical reports

---

### 🤖 Traffic Prediction (Machine Learning)

Uses **RandomForest Regressor** to predict traffic density for the next **24 hours**.

Features used:

* Hour of day
* Day of week
* Previous traffic patterns

---

### 📧 Email Alert System

Send automated alerts for:

* Mode changes (AUTO ↔ MANUAL)
* Camera inactivity
* Model training completion
* Prediction updates

---

### 📈 Web Dashboard

A real-time web interface built with **Flask + HTML + JavaScript** showing:

* Live video stream
* Traffic density
* Vehicle count
* Signal status
* Historical analytics
* Predictions

---

# 🏗 System Architecture

```
Camera / Video
       │
       ▼
YOLOv8 Vehicle Detection
       │
       ▼
Traffic Density Calculation
       │
       ▼
Signal Controller (AUTO/MANUAL)
       │
       ▼
SQLite Database Storage
       │
       ▼
Machine Learning Prediction
       │
       ▼
Flask Web Dashboard
```

---

# 📂 Project Structure

```
traffic-management-system/
│
├── app.py                  # Flask server & API routes
├── traffic_core.py         # Traffic detection & signal logic
├── database_manager.py     # SQLite database management
├── ml_model.py             # ML training & predictions
│
├── templates/
│   └── index.html          # Web dashboard
│
├── uploads/                # Uploaded videos
├── models/                 # Saved ML models
│
├── traffic_data.db         # SQLite database
├── requirements.txt        # Python dependencies
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Main libraries used:

* Flask
* OpenCV
* Ultralytics YOLOv8
* NumPy
* Pandas
* Scikit-learn
* ReportLab

---

### 3️⃣ Download YOLOv8 Model

The system automatically downloads:

```
yolov8n.pt
yolov8n-seg.pt
```

---

### 4️⃣ Run the Application

```
python app.py
```

---

# 🌐 Open the Web Dashboard

Open your browser and go to:

```
http://localhost:5000
```

---

# 🎮 How to Use

### Start Camera

Click **Start Camera** to begin real-time detection.

### Upload Traffic Video

Upload a traffic video to analyze traffic density.

### Switch Control Mode

* **AUTO** → AI controls signals
* **MANUAL** → User controls signals

### View Analytics

Access historical data and traffic statistics.

### Export Data

Export traffic reports in:

* **PDF**
* **CSV**
* **JSON**

---

# 📊 Database Tables

The system stores data in SQLite:

| Table            | Description           |
| ---------------- | --------------------- |
| traffic_history  | Traffic records       |
| sessions         | Application sessions  |
| mode_changes     | Mode switch logs      |
| manual_overrides | Manual signal actions |
| system_events    | System logs           |
| predictions      | ML predictions        |
| model_metadata   | Model training info   |
| email_settings   | Alert configuration   |

---

# 🧠 Machine Learning Model

Model Used:

```
RandomForestRegressor
```

Features:

* Hour of day
* Day of week
* Previous traffic counts

Output:

```
Predicted vehicle count
Predicted traffic density
Confidence score
```

---

# 📧 Email Alerts Configuration

Set environment variables:

```
export SMTP_USER=your_email@gmail.com
export SMTP_PASS=your_app_password
```

Or configure through the dashboard.

---

# 📦 API Endpoints

| Endpoint         | Description            |
| ---------------- | ---------------------- |
| `/start_camera`  | Start webcam detection |
| `/stop_camera`   | Stop detection         |
| `/upload`        | Upload traffic video   |
| `/video_feed`    | Live video stream      |
| `/stats`         | Current traffic stats  |
| `/history`       | Traffic history        |
| `/analytics`     | Traffic analytics      |
| `/predictions`   | ML predictions         |
| `/save_settings` | Update system settings |
| `/test_alert`    | Send test email alert  |

---

# 🧪 Technologies Used

* **Python**
* **Flask**
* **OpenCV**
* **YOLOv8**
* **NumPy**
* **Pandas**
* **Scikit-Learn**
* **SQLite**
* **ReportLab**

---

# 🎯 Future Improvements

* Multi-camera traffic monitoring
* Traffic violation detection
* License plate recognition
* Edge deployment (Raspberry Pi / Jetson)
* Smart city integration
* Deep learning traffic prediction (LSTM)

---

# 👨‍💻 Author

Developed as part of an **AI-based Smart Traffic Management System project**.

---

# 📜 License

This project is for **educational and research purposes**.

---

# ⭐ If you like this project

Give it a ⭐ on GitHub and contribute to improve the system!

