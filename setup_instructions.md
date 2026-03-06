# ⚙️ Setup Instructions

## AI Smart Traffic Management System

This guide explains how to install and run the **AI Smart Traffic Management System** on your computer.

---

# 1️⃣ Prerequisites

Before installing the project, make sure you have the following installed:

| Software | Version                  |
| -------- | ------------------------ |
| Python   | 3.9 or later             |
| pip      | Latest                   |
| Git      | Optional but recommended |

Check Python installation:

```bash
python --version
```

If Python is not installed, download it from:
https://www.python.org/downloads/

---

# 2️⃣ Clone the Repository

Clone the project using Git:

```bash
git clone https://github.com/your-username/ai-traffic-management-system.git
cd ai-traffic-management-system
```

Or download the project as a ZIP file and extract it.

---

# 3️⃣ Create a Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated.

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

# 4️⃣ Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Main libraries installed:

* Flask
* OpenCV
* Ultralytics YOLOv8
* NumPy
* Pandas
* Scikit-Learn
* ReportLab

---

# 5️⃣ Install PyTorch (Required for YOLOv8)

If PyTorch is not automatically installed, run:

### CPU Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### GPU Version (Optional)

Visit:

https://pytorch.org/get-started/locally/

and install the correct version for your CUDA.

---

# 6️⃣ Project Folder Structure

After installation your project should look like this:

```
traffic-management-system/
│
├── app.py
├── traffic_core.py
├── database_manager.py
├── ml_model.py
│
├── requirements.txt
├── README.md
├── setup_instructions.md
│
├── templates/
│   └── index.html
│
├── uploads/
├── models/
└── traffic_data.db
```

---

# 7️⃣ Run the Application

Start the Flask server:

```bash
python app.py
```

If successful you will see:

```
Running on http://127.0.0.1:5000
```

---

# 8️⃣ Open the Web Dashboard

Open your browser and go to:

```
http://localhost:5000
```

You will see the **Traffic Management Dashboard**.

---

# 9️⃣ Using the System

### Start Camera

Click **Start Camera** to analyze live traffic.

### Upload Video

Upload a traffic video file to analyze recorded traffic.

### Control Traffic Lights

Two modes are available:

AUTO Mode
AI automatically adjusts traffic signals.

MANUAL Mode
User controls signal states (Red / Yellow / Green).

---

# 🔟 Export Traffic Data

The system supports exporting reports in multiple formats:

* PDF
* CSV
* JSON

---

# 1️⃣1️⃣ Email Alerts (Optional)

To enable alerts, set environment variables:

```bash
export SMTP_USER=your_email@gmail.com
export SMTP_PASS=your_app_password
```

Or configure the email in the dashboard settings.

---

# 1️⃣2️⃣ Troubleshooting

### Camera Not Working

Check camera availability:

```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Module Not Found Error

Install dependencies again:

```bash
pip install -r requirements.txt
```

### YOLO Model Download Issues

Manually download models:

```
yolov8n.pt
yolov8n-seg.pt
```

Place them in the project root.

---

# 1️⃣3️⃣ Stop the Application

Press:

```
CTRL + C
```

in the terminal to stop the Flask server.

---

# 🎉 Setup Complete

Your **AI Smart Traffic Management System** is now ready to run.
