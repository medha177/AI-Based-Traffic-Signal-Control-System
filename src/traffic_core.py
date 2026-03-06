"""
AI Smart Traffic Management System - Traffic Core Module
Contains TrafficState class, signal controller logic, density calculation,
frame processing functions, and image enhancement
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from collections import defaultdict
import logging
import os
from datetime import datetime
import gc
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email configuration - NOW USING ENVIRONMENT VARIABLES
EMAIL_CONFIG = {
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'SMTP_USERNAME': os.getenv('SMTP_USER', 'reddymedha264@gmail.com'),
    'SMTP_PASSWORD': os.getenv('SMTP_PASS', 'uchujedysymwccor'),
    'SMTP_FROM': os.getenv('SMTP_USER', 'reddymedha264@gmail.com'),
    'ALERT_ENABLED': False,
    'ALERT_EMAIL': ''
}

# Database path - will be set from database_manager
LOCAL_DB_PATH = None

# Global configuration with default values
class Config:
    # Detection settings
    FRAME_SKIP = 2
    ROAD_MASK_UPDATE_FREQ = 10
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    # Data storage
    DATA_STORAGE_INTERVAL = 300  # 5 minutes in seconds
    FORCE_STORAGE_EVERY_N_FRAMES = 100
    
    # Training settings
    AUTO_TRAINING_INTERVAL = 3600  # 1 hour in seconds
    MIN_TRAINING_SAMPLES = 10
    TRAINING_TIME_LIMIT = 300
    PREDICTION_MODEL = 'random_forest'
    
    # Traffic light durations (seconds) - per density level
    SIGNAL_DURATIONS = {
        'RED': {'LOW': 30, 'MEDIUM': 15, 'HIGH': 10},
        'GREEN': {'LOW': 10, 'MEDIUM': 15, 'HIGH': 30},
        'YELLOW': 5
    }
    
    # Alert settings
    ALERT_TYPES = {
        'mode_change_auto_to_manual': True,
        'mode_change_manual_to_auto': True,
        'camera_inactive': True,
        'training_complete': True,
        'model_update': True
    }
    
    # Camera settings
    CAMERA_INACTIVITY_TIMEOUT = 300

# Global state management with thread safety
class TrafficState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.processing_active = False
        self.current_frame = None
        self.processed_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_counter = 0
        self.current_mode = "STANDBY"
        
        # Traffic Control Mode
        self.control_mode = "AUTO"
        self.manual_signal_state = "RED"
        
        # Vehicle counts
        self.total_vehicles = 0
        self.vehicle_counts = defaultdict(int)
        self.density_level = "LOW"
        
        # Signal state
        self.signal_state = "RED"
        self.signal_timer = 20
        self.base_duration = 20
        self.last_switch_time = time.time()
        self.green_start_time = None
        self.min_green_time = 10
        
        # Detection models
        self.det_model = None
        self.seg_model = None
        self.road_mask = None
        self.last_road_update = 0
        
        # Processing thread
        self.processing_thread = None
        # Signal controller thread
        self.signal_thread = None
        self.signal_running = True
        
        # Training thread
        self.training_thread = None
        self.training_active = False
        self.last_training_time = 0
        self.training_queue = None
        self.prediction_model = None
        self.model_accuracy = 0
        self.model_samples = 0
        self.current_predictions = []
        self.hourly_analysis = {}
        self.peak_hours = []
        
        # Camera inactivity tracking
        self.last_frame_time = time.time()
        self.camera_inactive_alert_sent = False
        
        # Data storage
        self.last_storage_time = time.time()
        self.last_frame_storage = 0
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Frame generator tracking
        self.frame_generator_active = False
        self.should_reset_generator = False
        self.last_frame_id = 0
        
        # Session initialization will be done after database is ready
        self.session_initialized = False

    def init_session(self):
        """Initialize session in database - called after database is ready"""
        global LOCAL_DB_PATH
        if LOCAL_DB_PATH is None:
            from database_manager import LOCAL_DB_PATH as DB_PATH
            globals()['LOCAL_DB_PATH'] = DB_PATH
        
        try:
            import sqlite3
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time, status, control_mode) VALUES (?, ?, ?, ?)",
                (self.session_id, datetime.now().isoformat(), 'active', self.control_mode)
            )
            conn.commit()
            conn.close()
            self.session_initialized = True
            logger.info(f"Session initialized: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")

    def log_event(self, event_type, description):
        """Log system event to database"""
        # Import here to avoid circular imports
        try:
            from database_manager import log_system_event as db_log_event
            db_log_event(event_type, description)
        except:
            pass

    def check_camera_inactivity(self):
        """Check if camera has been inactive and send alert if needed"""
        if not EMAIL_CONFIG['ALERT_ENABLED'] or not EMAIL_CONFIG['ALERT_EMAIL'] or not Config.ALERT_TYPES.get('camera_inactive', False):
            return
        
        if self.current_mode in ['CAMERA', 'VIDEO'] and self.processing_active:
            current_time = time.time()
            if current_time - self.last_frame_time > Config.CAMERA_INACTIVITY_TIMEOUT:
                if not self.camera_inactive_alert_sent:
                    send_alert(
                        "Camera Inactivity Alert",
                        f"Camera has been inactive for {Config.CAMERA_INACTIVITY_TIMEOUT/60:.0f} minutes. "
                        f"Last frame received: {datetime.fromtimestamp(self.last_frame_time).strftime('%H:%M:%S')}",
                        'camera_inactive'
                    )
                    self.camera_inactive_alert_sent = True
            else:
                self.camera_inactive_alert_sent = False

traffic_state = TrafficState()

# ==================== Helper Functions ====================

def validate_email(email):
    """
    Validate email format
    Returns (is_valid, error_message)
    """
    if not email or not isinstance(email, str):
        return False, "Email is required"
    
    email = email.strip()
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    # Additional checks for common email providers
    domain = email.split('@')[1].lower()
    common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    
    # Check if it's a common domain or has valid structure
    if domain not in common_domains and len(domain.split('.')) < 2:
        return False, "Invalid email domain"
    
    return True, "Valid email"

def send_alert(subject, message, alert_type=None):
    """
    Send email alert if enabled
    Returns (success, error_message)
    """
    logger.info(f"=== SEND ALERT ATTEMPT ===")
    logger.info(f"Subject: {subject}")
    logger.info(f"Alert Type: {alert_type}")
    logger.info(f"ALERT_ENABLED: {EMAIL_CONFIG['ALERT_ENABLED']}")
    logger.info(f"ALERT_EMAIL: '{EMAIL_CONFIG['ALERT_EMAIL']}'")
    
    # Check if email is configured
    if not EMAIL_CONFIG['ALERT_EMAIL']:
        error_msg = "Alert not sent: No alert email configured"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check if alerts are enabled globally
    if not EMAIL_CONFIG['ALERT_ENABLED']:
        error_msg = "Alert not sent: Alerts are disabled globally"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check if this specific alert type is enabled
    if alert_type and not Config.ALERT_TYPES.get(alert_type, False):
        error_msg = f"Alert not sent: Alert type '{alert_type}' is disabled"
        logger.info(error_msg)
        return False, error_msg
    
    # Validate email format
    is_valid, validation_error = validate_email(EMAIL_CONFIG['ALERT_EMAIL'])
    if not is_valid:
        error_msg = f"Alert not sent: Invalid email format - {validation_error}"
        logger.error(error_msg)
        return False, error_msg
    
    # Check SMTP credentials
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    logger.info(f"SMTP Configured: {smtp_configured}")
    
    if not smtp_configured:
        error_msg = "Alert not sent: SMTP credentials not configured"
        logger.error(error_msg)
        return False, error_msg
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['SMTP_FROM']
        msg['To'] = EMAIL_CONFIG['ALERT_EMAIL']
        msg['Subject'] = f"[Traffic Management] {subject}"
        
        body = f"""
        <html>
        <body>
            <h2>Traffic Management System Alert</h2>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Session:</strong> {traffic_state.session_id}</p>
            <hr>
            <p>{message}</p>
            <hr>
            <p><small>This is an automated message from your AI Traffic Management System</small></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server with timeout
        logger.info(f"Connecting to SMTP server {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']}")
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=30)
        
        # Start TLS
        logger.info("Starting TLS...")
        server.starttls()
        
        # Login
        logger.info(f"Attempting login with username: {EMAIL_CONFIG['SMTP_USERNAME']}")
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        
        # Send email
        logger.info(f"Sending email to {EMAIL_CONFIG['ALERT_EMAIL']}...")
        server.send_message(msg)
        
        # Close connection
        server.quit()
        
        success_msg = f"Alert sent successfully: {subject} to {EMAIL_CONFIG['ALERT_EMAIL']}"
        logger.info(success_msg)
        traffic_state.log_event("ALERT", f"Sent alert: {subject}")
        return True, success_msg
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication Failed: Check username/password - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except smtplib.SMTPException as e:
        error_msg = f"SMTP Error: {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except ConnectionRefusedError as e:
        error_msg = f"Connection Refused: Cannot connect to SMTP server {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']} - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except TimeoutError as e:
        error_msg = f"Connection Timeout: SMTP server not responding - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error sending alert: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg

def load_settings():
    """Load settings from database"""
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        
        # Load email settings
        cursor.execute("SELECT alert_email, alert_enabled FROM email_settings ORDER BY updated_at DESC LIMIT 1")
        email_row = cursor.fetchone()
        if email_row and email_row[0]:
            # Validate email before loading
            is_valid, _ = validate_email(email_row[0])
            if is_valid:
                EMAIL_CONFIG['ALERT_EMAIL'] = email_row[0]
                EMAIL_CONFIG['ALERT_ENABLED'] = bool(email_row[1]) if email_row[1] is not None else False
            else:
                logger.warning(f"Invalid email in database: {email_row[0]}")
                EMAIL_CONFIG['ALERT_EMAIL'] = ''
                EMAIL_CONFIG['ALERT_ENABLED'] = False
        
        conn.close()
        
        # Update Config class with loaded settings
        for key, value in rows:
            if key == 'SIGNAL_DURATIONS':
                Config.SIGNAL_DURATIONS = json.loads(value)
            elif key == 'ALERT_TYPES':
                Config.ALERT_TYPES = json.loads(value)
            elif hasattr(Config, key):
                if key in ['DATA_STORAGE_INTERVAL', 'CAMERA_INACTIVITY_TIMEOUT', 
                          'AUTO_TRAINING_INTERVAL', 'TRAINING_TIME_LIMIT', 'MIN_TRAINING_SAMPLES']:
                    setattr(Config, key, int(value))
                elif key == 'PREDICTION_MODEL':
                    setattr(Config, key, value)
                else:
                    setattr(Config, key, value)
        
        logger.info(f"Settings loaded from database. Alert email: {EMAIL_CONFIG['ALERT_EMAIL']}, Enabled: {EMAIL_CONFIG['ALERT_ENABLED']}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")

def save_settings():
    """Save current settings to database"""
    try:
        import sqlite3
        import json
        from datetime import datetime
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        settings = [
            ('DATA_STORAGE_INTERVAL', str(Config.DATA_STORAGE_INTERVAL)),
            ('AUTO_TRAINING_INTERVAL', str(Config.AUTO_TRAINING_INTERVAL)),
            ('TRAINING_TIME_LIMIT', str(Config.TRAINING_TIME_LIMIT)),
            ('MIN_TRAINING_SAMPLES', str(Config.MIN_TRAINING_SAMPLES)),
            ('PREDICTION_MODEL', Config.PREDICTION_MODEL),
            ('CAMERA_INACTIVITY_TIMEOUT', str(Config.CAMERA_INACTIVITY_TIMEOUT)),
            ('SIGNAL_DURATIONS', json.dumps(Config.SIGNAL_DURATIONS)),
            ('ALERT_TYPES', json.dumps(Config.ALERT_TYPES))
        ]
        
        for key, value in settings:
            cursor.execute('''
                INSERT INTO settings (key, value, updated_at) 
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
            ''', (key, value, datetime.now().isoformat(), value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.info("Settings saved to database")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

def save_email_settings(alert_email, alert_enabled):
    """
    Save email settings to database with validation
    Returns (success, message)
    """
    try:
        import sqlite3
        from datetime import datetime
        
        # Validate email if provided
        if alert_email:
            is_valid, validation_message = validate_email(alert_email)
            if not is_valid:
                logger.error(f"Email validation failed: {validation_message}")
                return False, validation_message
        
        # Clean and strip email
        alert_email = alert_email.strip() if alert_email else ''
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        # Ensure alert_enabled is stored as integer 1/0
        enabled_int = 1 if alert_enabled else 0
        
        cursor.execute('''
            INSERT INTO email_settings (alert_email, alert_enabled, updated_at)
            VALUES (?, ?, ?)
        ''', (alert_email, enabled_int, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Update runtime configuration
        EMAIL_CONFIG['ALERT_EMAIL'] = alert_email
        EMAIL_CONFIG['ALERT_ENABLED'] = alert_enabled
        
        logger.info(f"Email settings saved: '{alert_email}', Enabled: {alert_enabled} (stored as {enabled_int})")
        return True, "Email settings saved successfully"
        
    except Exception as e:
        error_msg = f"Failed to save email settings: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def update_config_from_settings(settings_data):
    """Update Config class with new settings"""
    try:
        # Update storage settings
        if 'storageInterval' in settings_data:
            Config.DATA_STORAGE_INTERVAL = int(settings_data['storageInterval'])
        
        if 'trainingInterval' in settings_data:
            Config.AUTO_TRAINING_INTERVAL = int(settings_data['trainingInterval'])
        
        if 'trainingTimeLimit' in settings_data:
            Config.TRAINING_TIME_LIMIT = int(settings_data['trainingTimeLimit'])
        
        if 'predictionModel' in settings_data:
            Config.PREDICTION_MODEL = settings_data['predictionModel']
        
        if 'cameraTimeout' in settings_data:
            Config.CAMERA_INACTIVITY_TIMEOUT = int(settings_data['cameraTimeout'])
        
        # Update signal durations
        if 'signalDurations' in settings_data:
            signal_data = settings_data['signalDurations']
            Config.SIGNAL_DURATIONS = {
                'RED': {
                    'LOW': int(signal_data['red']['low']),
                    'MEDIUM': int(signal_data['red']['medium']),
                    'HIGH': int(signal_data['red']['high'])
                },
                'GREEN': {
                    'LOW': int(signal_data['green']['low']),
                    'MEDIUM': int(signal_data['green']['medium']),
                    'HIGH': int(signal_data['green']['high'])
                },
                'YELLOW': int(signal_data['yellow'])
            }
        
        # Update alert types
        if 'alertTypes' in settings_data:
            alert_data = settings_data['alertTypes']
            Config.ALERT_TYPES = {
                'mode_change_auto_to_manual': alert_data.get('modeAutoManual', True),
                'mode_change_manual_to_auto': alert_data.get('modeManualAuto', True),
                'camera_inactive': alert_data.get('cameraInactive', True),
                'training_complete': alert_data.get('training', True),
                'model_update': alert_data.get('modelUpdate', True)
            }
        
        # Save to database
        save_settings()
        
        logger.info("Configuration updated successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False

def load_settings_endpoint():
    """Load settings from database and return them as dict"""
    try:
        settings_data = {
            'storageInterval': Config.DATA_STORAGE_INTERVAL,
            'trainingInterval': Config.AUTO_TRAINING_INTERVAL,
            'trainingTimeLimit': Config.TRAINING_TIME_LIMIT,
            'predictionModel': Config.PREDICTION_MODEL,
            'signalDurations': {
                'red': {
                    'low': Config.SIGNAL_DURATIONS['RED']['LOW'],
                    'medium': Config.SIGNAL_DURATIONS['RED']['MEDIUM'],
                    'high': Config.SIGNAL_DURATIONS['RED']['HIGH']
                },
                'green': {
                    'low': Config.SIGNAL_DURATIONS['GREEN']['LOW'],
                    'medium': Config.SIGNAL_DURATIONS['GREEN']['MEDIUM'],
                    'high': Config.SIGNAL_DURATIONS['GREEN']['HIGH']
                },
                'yellow': Config.SIGNAL_DURATIONS['YELLOW']
            },
            'alertTypes': {
                'modeAutoManual': Config.ALERT_TYPES.get('mode_change_auto_to_manual', True),
                'modeManualAuto': Config.ALERT_TYPES.get('mode_change_manual_to_auto', True),
                'cameraInactive': Config.ALERT_TYPES.get('camera_inactive', True),
                'training': Config.ALERT_TYPES.get('training_complete', True),
                'modelUpdate': Config.ALERT_TYPES.get('model_update', True)
            },
            'cameraTimeout': Config.CAMERA_INACTIVITY_TIMEOUT,
            'cameraIndex': 0,
            'frameSkip': Config.FRAME_SKIP,
            'detectionConfidence': 0.25,
            'alertEmail': EMAIL_CONFIG['ALERT_EMAIL'],
            'alertEnabled': EMAIL_CONFIG['ALERT_ENABLED']
        }
        return settings_data
    except Exception as e:
        logger.error(f"Failed to load settings endpoint: {e}")
        return {'error': str(e)}

def reset_for_new_input():
    """
    Completely stop previous processing and clear all data for new input
    Signal controller continues running independently
    """
    with traffic_state.lock:
        logger.info("=== RESETTING FOR NEW INPUT ===")
        
        # Step 1: Signal frame generator to reset
        traffic_state.should_reset_generator = True
        traffic_state.frame_generator_active = False
        
        # Step 2: Stop processing thread
        if traffic_state.processing_active:
            logger.info("Stopping active processing thread...")
            traffic_state.processing_active = False
            
            # Wait for thread to finish (with timeout)
            if traffic_state.processing_thread and traffic_state.processing_thread.is_alive():
                traffic_state.processing_thread.join(timeout=2.0)
                if traffic_state.processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")
        
        # Step 3: Release camera/video capture
        if traffic_state.camera:
            logger.info("Releasing camera/video capture...")
            traffic_state.camera.release()
            traffic_state.camera = None
        
        # Step 4: Clear all frames from memory
        logger.info("Clearing frame buffers...")
        traffic_state.current_frame = None
        traffic_state.processed_frame = None
        traffic_state.last_frame_id = 0
        
        # Step 5: Reset all vehicle data
        logger.info("Resetting vehicle data...")
        traffic_state.total_vehicles = 0
        traffic_state.vehicle_counts = defaultdict(int)
        traffic_state.density_level = "LOW"
        
        # Step 6: Reset frame counters
        traffic_state.frame_count = 0
        traffic_state.frame_counter = 0
        traffic_state.last_fps_update = time.time()
        traffic_state.last_frame_time = time.time()
        
        # Step 7: Reset road mask
        traffic_state.road_mask = None
        traffic_state.last_road_update = 0
        
        # Step 8: Force garbage collection of frames
        gc.collect()
        
        # Step 9: Small delay to ensure everything is cleaned up
        time.sleep(0.5)
        
        logger.info("=== RESET COMPLETE - SIGNAL CONTINUES ===")
        traffic_state.log_event("INFO", "System reset for new input completed")

def load_models():
    """Load YOLO models with error handling and far vehicle detection optimizations"""
    try:
        # Load detection model with optimized settings for far vehicles
        traffic_state.det_model = YOLO('yolov8n.pt')
        
        # Adjust detection parameters for better far vehicle detection
        traffic_state.det_model.conf = 0.25
        traffic_state.det_model.iou = 0.45
        traffic_state.det_model.max_det = 300
        
        # Load segmentation model
        traffic_state.seg_model = YOLO('yolov8n-seg.pt')
        
        logger.info("Models loaded successfully with far-vehicle optimizations")
        traffic_state.log_event("INFO", "Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        traffic_state.log_event("ERROR", f"Model loading failed: {e}")
        return False

def calculate_fps():
    """Calculate frames per second"""
    traffic_state.frame_counter += 1
    current_time = time.time()
    
    if current_time - traffic_state.last_fps_update >= 1.0:
        traffic_state.fps = traffic_state.frame_counter
        traffic_state.frame_counter = 0
        traffic_state.last_fps_update = current_time
    
    return traffic_state.fps

def get_signal_duration(density, signal_type):
    """Get signal duration based on traffic density from config"""
    if signal_type == "YELLOW":
        return Config.SIGNAL_DURATIONS.get('YELLOW', 5)
    else:
        return Config.SIGNAL_DURATIONS.get(signal_type, {}).get(density, 20)

def update_signal_state():
    """
    State machine for traffic signal control
    Runs differently based on AUTO/MANUAL mode
    """
    with traffic_state.lock:
        current_time = time.time()
        
        # If in MANUAL mode, don't auto-switch signals
        if traffic_state.control_mode == "MANUAL":
            # In manual mode, signal state is controlled by user
            # Just update the timer display
            if traffic_state.signal_state != traffic_state.manual_signal_state:
                traffic_state.signal_state = traffic_state.manual_signal_state
                traffic_state.last_switch_time = current_time
                # Set a fixed timer for manual mode (just for display)
                if traffic_state.signal_state == "RED":
                    traffic_state.signal_timer = 30
                elif traffic_state.signal_state == "YELLOW":
                    traffic_state.signal_timer = 5
                elif traffic_state.signal_state == "GREEN":
                    traffic_state.signal_timer = 30
            return
        
        # AUTO mode logic continues below
        elapsed = current_time - traffic_state.last_switch_time
        
        # Check if minimum green time has been satisfied
        if (traffic_state.signal_state == "GREEN" and 
            traffic_state.green_start_time and 
            current_time - traffic_state.green_start_time < traffic_state.min_green_time):
            return
        
        # Check if it's time to switch to next state
        if elapsed >= traffic_state.signal_timer:
            # State transition logic
            if traffic_state.signal_state == "RED":
                traffic_state.signal_state = "GREEN"
                traffic_state.green_start_time = current_time
                traffic_state.base_duration = get_signal_duration(
                    traffic_state.density_level, 'GREEN'
                )
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"[AUTO] Signal switched to GREEN, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            elif traffic_state.signal_state == "GREEN":
                traffic_state.signal_state = "YELLOW"
                traffic_state.base_duration = get_signal_duration(None, 'YELLOW')
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"[AUTO] Signal switched to YELLOW, duration: {traffic_state.signal_timer}s")
                
            elif traffic_state.signal_state == "YELLOW":
                traffic_state.signal_state = "RED"
                traffic_state.base_duration = get_signal_duration(
                    traffic_state.density_level, 'RED'
                )
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"[AUTO] Signal switched to RED, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            traffic_state.last_switch_time = current_time

def signal_controller_loop():
    """
    Dedicated thread for signal control
    Runs forever independent of detection
    """
    logger.info("Signal controller thread started - continuous signal simulation")
    traffic_state.log_event("INFO", "Signal controller thread started")
    
    storage_counter = 0
    training_check_counter = 0
    
    while traffic_state.signal_running:
        update_signal_state()
        
        # Check camera inactivity
        traffic_state.check_camera_inactivity()
        
        # Store data periodically (configurable interval)
        with traffic_state.lock:
            current_time = time.time()
            if current_time - traffic_state.last_storage_time >= Config.DATA_STORAGE_INTERVAL:
                # Import here to avoid circular imports
                try:
                    from database_manager import store_traffic_data
                    store_traffic_data()
                except:
                    pass
                traffic_state.last_storage_time = current_time
                storage_counter += 1
                logger.info(f"Periodic storage #{storage_counter} completed")
        
        # Check if it's time to train model
        training_check_counter += 1
        if (training_check_counter % 10 == 0 and
            not traffic_state.training_active and
            time.time() - traffic_state.last_training_time > Config.AUTO_TRAINING_INTERVAL):
            
            logger.info("Auto-training timer triggered")
            try:
                from ml_model import train_prediction_model
                traffic_state.training_thread = threading.Thread(target=train_prediction_model)
                traffic_state.training_thread.daemon = True
                traffic_state.training_thread.start()
            except:
                pass
        
        time.sleep(0.1)
    
    logger.info("Signal controller thread stopped")
    traffic_state.log_event("INFO", "Signal controller thread stopped")

def refresh_signal_controller():
    """
    Refresh the signal controller timers with updated settings
    Called after settings are saved to apply new durations immediately
    Does not stop or restart the signal thread
    """
    with traffic_state.lock:
        logger.info("=== REFRESHING SIGNAL CONTROLLER WITH NEW SETTINGS ===")
        
        # Store current state before refresh
        current_state = traffic_state.signal_state
        current_density = traffic_state.density_level
        
        # Recalculate base duration using updated Config.SIGNAL_DURATIONS
        traffic_state.base_duration = get_signal_duration(
            current_density,
            current_state
        )
        
        # Reset timer and switch time
        traffic_state.signal_timer = traffic_state.base_duration
        traffic_state.last_switch_time = time.time()
        
        # If in MANUAL mode, also update manual signal state
        if traffic_state.control_mode == "MANUAL":
            traffic_state.manual_signal_state = current_state
            logger.info(f"Manual mode: updated manual signal to {current_state}")
        
        logger.info(
            f"Signal controller refreshed - "
            f"State: {current_state}, "
            f"Density: {current_density}, "
            f"New duration: {traffic_state.base_duration}s, "
            f"Timer reset to: {traffic_state.signal_timer}s"
        )
        
        # Log the refresh event
        traffic_state.log_event(
            "SETTINGS_UPDATE",
            f"Signal timers refreshed - {current_state} duration set to {traffic_state.base_duration}s"
        )

def update_density_level():
    """
    Update traffic density based on vehicle count
    Dynamically updates signal duration WITHOUT resetting the cycle
    Only affects AUTO mode
    """
    with traffic_state.lock:
        old_density = traffic_state.density_level
        
        if traffic_state.total_vehicles <= 7:
            new_density = "LOW"
        elif 8 <= traffic_state.total_vehicles <= 15:
            new_density = "MEDIUM"
        else:  # vehicles > 15
            new_density = "HIGH"
        
        # Update density level
        traffic_state.density_level = new_density
        
        # Only auto-adjust timings in AUTO mode
        if traffic_state.control_mode == "AUTO":
            # If density changed and we're in RED or GREEN (not YELLOW)
            if old_density != new_density and traffic_state.signal_state in ['RED', 'GREEN']:
                # Calculate elapsed time in current state
                current_time = time.time()
                elapsed = current_time - traffic_state.last_switch_time
                
                # Get new base duration for current signal based on new density
                new_duration = get_signal_duration(traffic_state.density_level, traffic_state.signal_state)
                old_duration = traffic_state.base_duration
                
                if new_duration != old_duration:
                    # Store the elapsed time proportion
                    progress_ratio = elapsed / old_duration if old_duration > 0 else 0
                    
                    # Update base duration
                    traffic_state.base_duration = new_duration
                    
                    # Calculate new remaining time based on progress
                    remaining = max(1, new_duration * (1 - min(progress_ratio, 0.95)))
                    traffic_state.signal_timer = remaining
                    
                    logger.info(f"Density changed: {old_density}->{new_density}: "
                              f"Updated {traffic_state.signal_state} duration to {new_duration}s, "
                              f"remaining: {remaining:.1f}s")
            
            elif old_density != new_density:
                logger.info(f"Density changed from {old_density} to {new_density} (vehicles: {traffic_state.total_vehicles})")

def enhance_image_for_far_vehicles(frame):
    """Apply image preprocessing to improve far vehicle detection"""
    try:
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # Blend with original for natural look
        result = cv2.addWeighted(frame, 0.5, sharpened, 0.5, 0)
        
        return result
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return frame

def get_road_mask(frame):
    """Extract road mask using segmentation model"""
    if traffic_state.frame_count % Config.ROAD_MASK_UPDATE_FREQ == 0:
        results = traffic_state.seg_model(frame, verbose=False)
        
        if results[0].masks is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for seg_mask in results[0].masks.data:
                seg_mask = seg_mask.cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                mask = cv2.bitwise_or(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
            
            traffic_state.road_mask = mask
        else:
            traffic_state.road_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    return traffic_state.road_mask

def process_frame(frame):
    """
    Main processing pipeline for each frame
    ONLY does detection - does NOT control signal
    """
    if frame is None:
        return None
    
    # Update last frame time for inactivity detection
    traffic_state.last_frame_time = time.time()
    
    # Calculate FPS
    calculate_fps()
    
    # Skip frames for performance
    traffic_state.frame_count += 1
    if traffic_state.frame_count % Config.FRAME_SKIP != 0:
        return traffic_state.processed_frame if traffic_state.processed_frame is not None else frame
    
    # Enhance image for far vehicle detection
    enhanced_frame = enhance_image_for_far_vehicles(frame)
    
    # Get road mask
    road_mask = get_road_mask(enhanced_frame)
    
    if road_mask is None:
        return frame
    
    # Run vehicle detection with optimized settings for far vehicles
    results = traffic_state.det_model(
        enhanced_frame, 
        classes=list(Config.VEHICLE_CLASSES.keys()), 
        verbose=False,
        conf=0.25,
        iou=0.45
    )
    
    # Reset counts for this frame
    current_counts = defaultdict(int)
    total_valid = 0
    
    # Create overlay frame
    overlay = frame.copy()
    
    # Process detections
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if 0 <= center_y < road_mask.shape[0] and 0 <= center_x < road_mask.shape[1]:
                if road_mask[center_y, center_x] > 0:
                    total_valid += 1
                    vehicle_type = Config.VEHICLE_CLASSES.get(int(cls), 'unknown')
                    current_counts[vehicle_type] += 1
                    
                    # Color based on confidence (brighter for higher confidence)
                    color_intensity = int(conf * 255)
                    color = (0, color_intensity, 255 - color_intensity)
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Add confidence to label for far vehicles
                    box_size = (x2 - x1) * (y2 - y1)
                    if box_size < 5000:  # Small box = far vehicle
                        label = f"{vehicle_type} ({conf:.2f})"
                    else:
                        label = vehicle_type
                    
                    cv2.putText(overlay, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.circle(overlay, (center_x, center_y), 2, (0, 0, 255), -1)
    
    # Update global counts (thread-safe)
    with traffic_state.lock:
        traffic_state.total_vehicles = total_valid
        traffic_state.vehicle_counts = dict(current_counts)
    
    # Update density level
    update_density_level()
    
    # Force store data periodically (configurable)
    if traffic_state.frame_count % Config.FORCE_STORAGE_EVERY_N_FRAMES == 0 and traffic_state.frame_count > 0:
        try:
            from database_manager import store_traffic_data
            store_traffic_data(force=True)
        except:
            pass
    
    # Add semi-transparent road mask overlay
    if road_mask is not None:
        road_colored = cv2.applyColorMap(road_mask, cv2.COLORMAP_BONE)
        overlay = cv2.addWeighted(overlay, 0.8, road_colored, 0.2, 0)
    
    # Draw traffic information overlay
    overlay = draw_info_overlay(overlay)
    
    return overlay

def draw_info_overlay(frame):
    """Draw traffic statistics and signal information on frame"""
    height, width = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (480, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    y_offset = 40
    line_height = 25
    
    # Mode indicator
    mode_color = (0, 255, 0) if traffic_state.control_mode == "AUTO" else (255, 255, 0)
    cv2.putText(frame, f"Mode: {traffic_state.control_mode}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Total Vehicles: {traffic_state.total_vehicles}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"FPS: {traffic_state.fps}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    for vtype, count in traffic_state.vehicle_counts.items():
        cv2.putText(frame, f"{vtype}: {count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    y_offset += 5
    
    density_color = {
        'LOW': (0, 255, 0),
        'MEDIUM': (0, 255, 255),
        'HIGH': (0, 0, 255)
    }.get(traffic_state.density_level, (255, 255, 255))
    
    cv2.putText(frame, f"Density: {traffic_state.density_level}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)
    y_offset += line_height
    
    signal_color = {
        'RED': (0, 0, 255),
        'YELLOW': (0, 255, 255),
        'GREEN': (0, 255, 0)
    }.get(traffic_state.signal_state, (255, 255, 255))
    
    cv2.putText(frame, f"Signal: {traffic_state.signal_state}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, signal_color, 2)
    y_offset += line_height
    
    remaining = max(0, traffic_state.signal_timer - (time.time() - traffic_state.last_switch_time))
    cv2.putText(frame, f"Timer: {remaining:.1f}s", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add storage info
    time_since_last = time.time() - traffic_state.last_storage_time
    next_storage = max(0, Config.DATA_STORAGE_INTERVAL - time_since_last)
    cv2.putText(frame, f"Next storage: {next_storage/60:.1f}min", 
                (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Add model info
    if traffic_state.model_accuracy > 0:
        cv2.putText(frame, f"Model acc: {traffic_state.model_accuracy:.1f}%", 
                   (20, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    # Add prediction info
    if traffic_state.current_predictions:
        next_hour = traffic_state.current_predictions[0] if traffic_state.current_predictions else None
        if next_hour:
            cv2.putText(frame, f"Next hour: {next_hour['density']} ({next_hour['predicted_count']} veh)", 
                       (20, y_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
    
    cv2.putText(frame, f"Session: {traffic_state.session_id[-8:]}", 
                (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.putText(frame, f"Input: {traffic_state.current_mode}", 
                (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Alert indicator
    if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL']:
        cv2.putText(frame, "Alerts ON", (width - 120, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Manual mode hint
    if traffic_state.control_mode == "MANUAL":
        cv2.putText(frame, "Manual Control - Use buttons", 
                   (width - 300, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw traffic light
    light_x = width - 80
    light_y = 40
    radius = 20
    
    cv2.rectangle(frame, (light_x - 25, light_y - 10), (light_x + 25, light_y + 70), (50, 50, 50), -1)
    
    red_color = (0, 0, 255) if traffic_state.signal_state == 'RED' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y), radius, red_color, -1)
    
    yellow_color = (0, 255, 255) if traffic_state.signal_state == 'YELLOW' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y + 30), radius, yellow_color, -1)
    
    green_color = (0, 255, 0) if traffic_state.signal_state == 'GREEN' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y + 60), radius, green_color, -1)
    
    return frame

def generate_frames():
    """Generator function for MJPEG streaming with reset capability"""
    logger.info("Frame generator started")
    traffic_state.frame_generator_active = True
    traffic_state.should_reset_generator = False
    
    frame_count = 0
    last_frame_id = 0
    
    while traffic_state.frame_generator_active:
        # Check if we need to reset
        if traffic_state.should_reset_generator:
            logger.info("Frame generator reset signal received")
            traffic_state.should_reset_generator = False
            break
            
        if traffic_state.processed_frame is not None:
            # Check if this is a new frame
            current_frame_id = id(traffic_state.processed_frame)
            if current_frame_id != last_frame_id:
                last_frame_id = current_frame_id
                ret, buffer = cv2.imencode('.jpg', traffic_state.processed_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_count += 1
                    
                    # Log every 100 frames
                    if frame_count % 100 == 0:
                        logger.info(f"Frame generator: sent {frame_count} frames")
        
        time.sleep(0.03)  # ~30 fps
    
    logger.info(f"Frame generator stopped after sending {frame_count} frames")
    traffic_state.frame_generator_active = False

def camera_processing_loop():
    """Main processing loop for camera/video input"""
    logger.info(f"Processing loop started for mode: {traffic_state.current_mode}")
    
    frame_counter = 0
    no_frame_count = 0
    
    while traffic_state.processing_active and traffic_state.camera and traffic_state.camera.isOpened():
        ret, frame = traffic_state.camera.read()
        if not ret:
            no_frame_count += 1
            logger.error(f"Failed to read frame from source (attempt {no_frame_count})")
            
            # If we've failed multiple times, assume the video ended
            if no_frame_count > 10:
                logger.info("Video source ended, stopping processing loop")
                break
            time.sleep(0.1)
            continue
        
        no_frame_count = 0  # Reset counter on successful read
        traffic_state.current_frame = frame
        processed = process_frame(frame)
        
        if processed is not None:
            traffic_state.processed_frame = processed
            frame_counter += 1
            
            # Log every 100 frames
            if frame_counter % 100 == 0:
                logger.info(f"Processed {frame_counter} frames, current vehicles: {traffic_state.total_vehicles}")
    
    logger.info(f"Processing loop ended after {frame_counter} frames")
    
    # Ensure processing is marked as inactive
    with traffic_state.lock:
        traffic_state.processing_active = False
        
    if traffic_state.camera:
        logger.info("Releasing camera capture...")
        traffic_state.camera.release()
        traffic_state.camera = None
    
    # Signal the frame generator to reset
    traffic_state.should_reset_generator = True
    
    # Log the end of processing
    traffic_state.log_event("INFO", f"Video processing ended, processed {frame_counter} frames")

def start_signal_controller():
    """Start the continuous signal controller thread"""
    traffic_state.signal_thread = threading.Thread(target=signal_controller_loop)
    traffic_state.signal_thread.daemon = True
    traffic_state.signal_thread.start()
    logger.info("Signal controller thread launched")

def reset_system():
    """Reset system but preserve signal state"""
    with traffic_state.lock:
        logger.info("Resetting system (signal continues)...")
        
        # Stop processing but keep signal running
        traffic_state.processing_active = False
        time.sleep(0.5)
        
        # Release camera/video capture
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
        
        # Clear frame buffers
        traffic_state.current_frame = None
        traffic_state.processed_frame = None
        traffic_state.frame_count = 0
        traffic_state.road_mask = None
        
        # Reset vehicle statistics
        traffic_state.total_vehicles = 0
        traffic_state.vehicle_counts = defaultdict(int)
        traffic_state.density_level = "LOW"
        
        # Reset mode
        traffic_state.current_mode = "STANDBY"
        
        logger.info("System reset complete - signal continues running")

def cleanup():
    """Cleanup function for graceful shutdown"""
    import time
    
    logger.info("Shutting down...")
    traffic_state.log_event("INFO", "System shutdown initiated")
    traffic_state.signal_running = False
    time.sleep(0.5)
    
    # Store final data point
    try:
        from database_manager import store_traffic_data, complete_session
        logger.info("Storing final data point...")
        store_traffic_data(force=True)
        
        # Update session status
        complete_session(traffic_state.session_id)
    except:
        pass
    
    reset_for_new_input()
    cv2.destroyAllWindows()
    logger.info("Cleanup complete")