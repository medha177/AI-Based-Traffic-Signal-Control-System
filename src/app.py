"""
AI Smart Traffic Management System - Main Application Entry Point
Flask routes and application initialization
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import threading
import time
import logging
import os
from datetime import datetime, timedelta
import atexit
import json
import gc
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import io
import csv
import re
from collections import defaultdict

# Import from our modules
from traffic_core import (
    traffic_state, TrafficState, Config, EMAIL_CONFIG,
    reset_for_new_input, load_models, calculate_fps, process_frame,
    generate_frames, camera_processing_loop, get_signal_duration,
    update_signal_state, signal_controller_loop, refresh_signal_controller,
    update_density_level, enhance_image_for_far_vehicles, get_road_mask,
    draw_info_overlay, start_signal_controller, reset_system, cleanup,
    validate_email, send_alert, load_settings, save_settings, save_email_settings,
    update_config_from_settings, load_settings_endpoint
)

from database_manager import (
    init_database, store_traffic_data, get_traffic_history,
    get_traffic_analytics, log_system_event, log_mode_change,
    log_manual_override, update_session_record_count, complete_session,
    LOCAL_DB_PATH
)

from ml_model import (
    train_prediction_model, generate_predictions, generate_demo_predictions,
    analyze_hourly_patterns, save_model_to_disk, load_saved_model,
    MODEL_PATH, MODEL_DIR
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with development configuration
app = Flask(__name__)
app.secret_key = 'traffic-management-secret-key-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize database
init_database()

# Load settings from database
load_settings()


# ==================== Flask Routes ====================

@app.after_request
def add_header(response):
    """Add headers to disable caching"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    """Set control mode to AUTO or MANUAL"""
    if mode.upper() not in ['AUTO', 'MANUAL']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    with traffic_state.lock:
        old_mode = traffic_state.control_mode
        traffic_state.control_mode = mode.upper()
        
        # If switching to MANUAL, set manual signal to current signal
        if mode.upper() == "MANUAL":
            traffic_state.manual_signal_state = traffic_state.signal_state
        
        logger.info(f"Control mode changed: {old_mode} -> {mode.upper()}")
        log_system_event("MODE_CHANGE", f"Mode changed from {old_mode} to {mode.upper()}")
        
        # Send alert if enabled
        alert_type = None
        if old_mode == "AUTO" and mode.upper() == "MANUAL":
            alert_type = 'mode_change_auto_to_manual'
            message = f"System switched from AUTO to MANUAL mode at {datetime.now().strftime('%H:%M:%S')}"
        elif old_mode == "MANUAL" and mode.upper() == "AUTO":
            alert_type = 'mode_change_manual_to_auto'
            message = f"System switched from MANUAL to AUTO mode at {datetime.now().strftime('%H:%M:%S')}"
        else:
            message = f"Mode changed from {old_mode} to {mode.upper()}"
        
        send_alert("Mode Change Alert", message, alert_type)
        
        # Store mode change
        log_mode_change(old_mode, mode.upper())
    
    return jsonify({
        'status': 'success',
        'mode': mode.upper(),
        'message': f'Switched to {mode.upper()} mode'
    })

@app.route('/set_signal/<state>')
def set_signal(state):
    """Manually set signal state (only works in MANUAL mode)"""
    if state.upper() not in ['RED', 'YELLOW', 'GREEN']:
        return jsonify({'error': 'Invalid signal state'}), 400
    
    with traffic_state.lock:
        if traffic_state.control_mode != "MANUAL":
            return jsonify({
                'error': 'Cannot manually set signal in AUTO mode',
                'current_mode': traffic_state.control_mode
            }), 400
        
        old_state = traffic_state.manual_signal_state
        traffic_state.manual_signal_state = state.upper()
        
        # Force immediate update
        traffic_state.signal_state = state.upper()
        traffic_state.last_switch_time = time.time()
        
        # Set appropriate timer for display
        if state.upper() == "RED":
            traffic_state.signal_timer = 30
        elif state.upper() == "YELLOW":
            traffic_state.signal_timer = 5
        elif state.upper() == "GREEN":
            traffic_state.signal_timer = 30
        
        logger.info(f"Manual signal changed: {old_state} -> {state.upper()}")
        log_system_event("MANUAL_OVERRIDE", f"Signal changed from {old_state} to {state.upper()}")
        
        # Store manual override
        log_manual_override(old_state, state.upper())
    
    return jsonify({
        'status': 'success',
        'signal': state.upper(),
        'message': f'Signal set to {state.upper()}'
    })

@app.route('/get_mode')
def get_mode():
    """Get current control mode"""
    with traffic_state.lock:
        return jsonify({
            'control_mode': traffic_state.control_mode,
            'signal_state': traffic_state.signal_state,
            'manual_signal': traffic_state.manual_signal_state if traffic_state.control_mode == "MANUAL" else None,
            'session_id': traffic_state.session_id
        })

@app.route('/start_camera')
def start_camera():
    """Start webcam processing - signal continues independently"""
    logger.info("Starting camera...")
    
    # COMPLETE RESET before starting new input
    reset_for_new_input()
    
    # Initialize camera
    traffic_state.camera = cv2.VideoCapture(0)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open camera")
        log_system_event("ERROR", "Failed to open camera")
        return jsonify({'error': 'Failed to open camera'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.current_mode = "CAMERA"
        logger.info(f"Processing active set to: {traffic_state.processing_active}")
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Camera started successfully - signal continues")
    log_system_event("INFO", "Camera started")
    return jsonify({'status': 'Camera started', 'mode': 'CAMERA'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera processing - signal continues running"""
    logger.info("Stopping camera...")
    
    with traffic_state.lock:
        traffic_state.processing_active = False
        traffic_state.current_mode = "STANDBY"
        
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
    
    logger.info("Camera stopped - signal continues")
    log_system_event("INFO", "Camera stopped")
    return jsonify({'status': 'Camera stopped', 'mode': 'STANDBY'})

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload - signal continues independently"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    logger.info(f"Processing uploaded video: {video_file.filename}")
    
    # COMPLETE RESET before starting new input
    reset_for_new_input()
    
    # Save uploaded video temporarily
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'uploads/traffic_{timestamp}.mp4'
    os.makedirs('uploads', exist_ok=True)
    video_file.save(video_path)
    
    # Initialize video capture
    traffic_state.camera = cv2.VideoCapture(video_path)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open video file")
        log_system_event("ERROR", f"Failed to open video file: {video_file.filename}")
        return jsonify({'error': 'Failed to open video file'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.current_mode = "VIDEO"
        logger.info(f"Processing active set to: {traffic_state.processing_active}")
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Video processing started - signal continues")
    log_system_event("INFO", f"Video processing started: {video_file.filename}")
    return jsonify({'status': 'Video processing started', 'mode': 'VIDEO'})

@app.route('/video_feed')
def video_feed():
    """MJPEG video feed endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """JSON endpoint for traffic statistics"""
    with traffic_state.lock:
        remaining = max(0, traffic_state.signal_timer - (time.time() - traffic_state.last_switch_time))
        stats = {
            'total_vehicles': traffic_state.total_vehicles,
            'vehicle_counts': dict(traffic_state.vehicle_counts),
            'density_level': traffic_state.density_level,
            'signal_state': traffic_state.signal_state,
            'signal_timer': round(remaining, 1),
            'base_duration': traffic_state.base_duration,
            'processing_active': traffic_state.processing_active,
            'current_mode': traffic_state.current_mode,
            'fps': traffic_state.fps,
            'control_mode': traffic_state.control_mode,
            'manual_signal': traffic_state.manual_signal_state if traffic_state.control_mode == "MANUAL" else None,
            'session_id': traffic_state.session_id,
            'model_accuracy': round(traffic_state.model_accuracy, 1) if traffic_state.model_accuracy > 0 else None,
            'training_active': traffic_state.training_active,
            'storage_interval_min': Config.DATA_STORAGE_INTERVAL / 60,
            'predictions_available': len(traffic_state.current_predictions) > 0,
            'alert_email': EMAIL_CONFIG['ALERT_EMAIL'] if EMAIL_CONFIG['ALERT_EMAIL'] else None,
            'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED']
        }
    return jsonify(stats)

@app.route('/history')
def get_history():
    """Get traffic history from database"""
    limit = request.args.get('limit', 100, type=int)
    session = request.args.get('session', None)
    history = get_traffic_history(limit, session)
    
    # Get total records count
    total_records = 0
    try:
        import sqlite3
        from database_manager import LOCAL_DB_PATH
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM traffic_history")
        total_records = cursor.fetchone()[0]
        conn.close()
    except:
        pass
    
    return jsonify({
        'history': history, 
        'count': len(history),
        'total_records': total_records
    })

@app.route('/analytics')
def get_analytics():
    """Get traffic analytics from database"""
    analytics = get_traffic_analytics()
    return jsonify({'analytics': analytics})

@app.route('/predictions')
def get_predictions():
    """Get traffic predictions from database"""
    try:
        from ml_model import generate_demo_predictions
        
        # First check if we have current predictions in memory
        if hasattr(traffic_state, 'current_predictions') and traffic_state.current_predictions:
            predictions = traffic_state.current_predictions
        else:
            # Try to get from database
            import sqlite3
            from database_manager import LOCAL_DB_PATH
            
            conn = sqlite3.connect(LOCAL_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get most recent predictions (one per hour)
            cursor.execute('''
                SELECT p1.* 
                FROM predictions p1
                INNER JOIN (
                    SELECT hour, MAX(timestamp) as max_timestamp
                    FROM predictions
                    GROUP BY hour
                ) p2 ON p1.hour = p2.hour AND p1.timestamp = p2.max_timestamp
                ORDER BY p1.hour ASC
                LIMIT 24
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in rows:
                pred = dict(row)
                # Format for frontend
                pred['time'] = f"{pred['hour']:02d}:00"
                if 'predicted_count' not in pred:
                    pred['predicted_count'] = 10
                predictions.append(pred)
        
        # If still no predictions, generate demo predictions
        if not predictions:
            predictions = generate_demo_predictions()
            traffic_state.current_predictions = predictions
        
        # Get hourly analysis
        hourly_analysis = traffic_state.hourly_analysis if hasattr(traffic_state, 'hourly_analysis') else {}
        
        # Get peak hours
        peak_hours = traffic_state.peak_hours if hasattr(traffic_state, 'peak_hours') else []
        
        return jsonify({
            'predictions': predictions,
            'hourly_analysis': hourly_analysis,
            'peak_hours': peak_hours,
            'model_accuracy': traffic_state.model_accuracy if traffic_state.model_accuracy > 0 else 85.0,
            'samples_used': traffic_state.model_samples if traffic_state.model_samples > 0 else 100,
            'last_training': datetime.fromtimestamp(traffic_state.last_training_time).isoformat() 
                            if traffic_state.last_training_time > 0 else None
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        # Return demo predictions even on error
        from ml_model import generate_demo_predictions
        predictions = generate_demo_predictions()
        
        return jsonify({
            'predictions': predictions,
            'hourly_analysis': {},
            'peak_hours': [],
            'model_accuracy': 85.0,
            'samples_used': 100,
            'last_training': datetime.now().isoformat()
        })

@app.route('/force_train')
def force_train():
    """Force model training immediately"""
    from ml_model import train_prediction_model
    import threading
    
    if traffic_state.training_active:
        return jsonify({'status': 'Training already in progress'})
    
    threading.Thread(target=train_prediction_model).start()
    return jsonify({'status': 'Training started'})

@app.route('/force_store')
def force_store():
    """Force store current data immediately"""
    success = store_traffic_data(force=True)
    return jsonify({
        'status': 'success' if success else 'error',
        'message': 'Data stored successfully' if success else 'Failed to store data',
        'vehicles': traffic_state.total_vehicles,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/db_stats')
def db_stats():
    """Get database statistics"""
    try:
        import sqlite3
        import os
        from database_manager import LOCAL_DB_PATH
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        # Get table sizes
        cursor.execute("SELECT COUNT(*) FROM traffic_history")
        history_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        sessions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mode_changes")
        mode_changes_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM manual_overrides")
        overrides_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM system_events")
        events_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_metadata")
        models_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT alert_email, alert_enabled FROM email_settings ORDER BY updated_at DESC LIMIT 1")
        email_row = cursor.fetchone()
        alert_email = email_row[0] if email_row else None
        alert_enabled = bool(email_row[1]) if email_row and len(email_row) > 1 else False
        
        # Get latest record
        cursor.execute("SELECT timestamp, vehicle_count FROM traffic_history ORDER BY timestamp DESC LIMIT 1")
        latest = cursor.fetchone()
        
        # Get database file size
        db_size = os.path.getsize(LOCAL_DB_PATH) if os.path.exists(LOCAL_DB_PATH) else 0
        
        conn.close()
        
        return jsonify({
            'traffic_history': history_count,
            'sessions': sessions_count,
            'mode_changes': mode_changes_count,
            'manual_overrides': overrides_count,
            'system_events': events_count,
            'predictions': predictions_count,
            'trained_models': models_count,
            'database_size_kb': round(db_size / 1024, 2),
            'database_path': os.path.abspath(LOCAL_DB_PATH),
            'alert_email': alert_email,
            'alert_enabled': alert_enabled,
            'latest_record': {
                'timestamp': latest[0] if latest else None,
                'vehicles': latest[1] if latest else None
            } if latest else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_email', methods=['POST'])
def save_email():
    """Save email settings with validation"""
    try:
        data = request.get_json()
        alert_email = data.get('alert_email', '').strip()
        alert_enabled = data.get('alert_enabled', False)
        
        logger.info(f"Saving email settings: email='{alert_email}', enabled={alert_enabled}")
        
        # Validate email if provided
        if alert_email:
            is_valid, validation_message = validate_email(alert_email)
            if not is_valid:
                logger.warning(f"Email validation failed: {validation_message}")
                return jsonify({
                    'status': 'error',
                    'message': validation_message
                }), 400
        
        # Save to database
        success, message = save_email_settings(alert_email, alert_enabled)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            }), 500
            
    except Exception as e:
        error_msg = f"Error saving email: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/save_settings', methods=['POST'])
def save_all_settings():
    """Save all settings from frontend"""
    try:
        data = request.get_json()
        
        # Store old signal durations for comparison
        old_red_durations = Config.SIGNAL_DURATIONS['RED'].copy()
        old_green_durations = Config.SIGNAL_DURATIONS['GREEN'].copy()
        old_yellow = Config.SIGNAL_DURATIONS['YELLOW']
        
        # Update Config class with new settings
        success = update_config_from_settings(data)
        
        # Get new signal durations for comparison
        new_red_durations = Config.SIGNAL_DURATIONS['RED']
        new_green_durations = Config.SIGNAL_DURATIONS['GREEN']
        new_yellow = Config.SIGNAL_DURATIONS['YELLOW']
        
        # Check if signal durations actually changed
        signal_durations_changed = (
            old_red_durations != new_red_durations or
            old_green_durations != new_green_durations or
            old_yellow != new_yellow
        )
        
        # Save email settings if provided
        if 'alertEmail' in data:
            alert_email = data.get('alertEmail', '').strip()
            alert_enabled = data.get('alertEnabled', False)
            
            # Validate email if provided
            if alert_email:
                is_valid, validation_message = validate_email(alert_email)
                if not is_valid:
                    logger.warning(f"Email validation failed: {validation_message}")
                    return jsonify({
                        'status': 'warning',
                        'message': f'Settings saved but email validation failed: {validation_message}'
                    })
            
            save_email_settings(alert_email, alert_enabled)
        
        if success:
            # If signal durations changed, refresh the signal controller
            if signal_durations_changed:
                logger.info("Signal durations changed - refreshing signal controller")
                refresh_signal_controller()
            else:
                logger.info("Settings saved but signal durations unchanged")
            
            return jsonify({
                'status': 'success', 
                'message': 'Settings saved successfully'
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to save settings'
            }), 500
            
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_settings')
def load_settings_endpoint():
    """Load settings from database and return them"""
    return jsonify(load_settings_endpoint())

@app.route('/test_alert')
def test_alert():
    """
    Send a test alert with detailed diagnostics
    Returns JSON with clear explanation of what happened
    """
    logger.info("=== TEST ALERT REQUESTED ===")
    
    # Check if email is configured
    if not EMAIL_CONFIG['ALERT_EMAIL']:
        logger.warning("Test alert failed: No alert email configured")
        return jsonify({
            'success': False,
            'message': 'No alert email configured',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': None,
                'smtp_configured': bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD']),
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200
    
    # Check if alerts are enabled globally
    if not EMAIL_CONFIG['ALERT_ENABLED']:
        logger.warning("Test alert failed: Alerts are disabled globally")
        return jsonify({
            'success': False,
            'message': 'Alerts are disabled globally',
            'details': {
                'alert_enabled': False,
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
            }
        }), 200
    
    # Check SMTP credentials
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    if not smtp_configured:
        logger.warning("Test alert failed: SMTP credentials not configured")
        return jsonify({
            'success': False,
            'message': 'SMTP credentials not configured',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': False,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200
    
    # Send test alert
    success, message = send_alert(
        "Test Alert",
        f"This is a test alert from your Traffic Management System.\n\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Session: {traffic_state.session_id}\n"
        f"Mode: {traffic_state.control_mode}\n"
        f"Vehicles: {traffic_state.total_vehicles}",
        None  # No alert type for test
    )
    
    if success:
        logger.info(f"Test alert sent successfully to {EMAIL_CONFIG['ALERT_EMAIL']}")
        return jsonify({
            'success': True,
            'message': f'Test alert sent successfully to {EMAIL_CONFIG["ALERT_EMAIL"]}',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': smtp_configured,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        })
    else:
        logger.error(f"Test alert failed: {message}")
        return jsonify({
            'success': False,
            'message': message,
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': smtp_configured,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200

@app.route('/email_status')
def email_status():
    """
    Debug endpoint to check email configuration status
    Returns JSON with current email settings and diagnostics
    """
    import smtplib
    
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    
    # Test SMTP connection if credentials are configured
    smtp_test_result = None
    if smtp_configured and EMAIL_CONFIG['ALERT_EMAIL']:
        try:
            # Quick SMTP connection test without sending email
            logger.info("Testing SMTP connection...")
            server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=10)
            server.starttls()
            server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
            server.quit()
            smtp_test_result = "Connection successful"
            logger.info("SMTP connection test successful")
        except Exception as e:
            smtp_test_result = f"Connection failed: {str(e)}"
            logger.error(f"SMTP connection test failed: {e}")
    
    return jsonify({
        'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
        'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
        'smtp_configured': smtp_configured,
        'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
        'smtp_port': EMAIL_CONFIG['SMTP_PORT'],
        'smtp_username': EMAIL_CONFIG['SMTP_USERNAME'][:3] + '...' if EMAIL_CONFIG['SMTP_USERNAME'] else None,
        'smtp_test': smtp_test_result,
        'alert_types': Config.ALERT_TYPES
    })

@app.route('/test_smtp')
def test_smtp():
    """
    Test SMTP connection without sending email
    Useful for debugging connection issues
    """
    import smtplib
    
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    
    if not smtp_configured:
        return jsonify({
            'success': False,
            'message': 'SMTP credentials not configured',
            'smtp_configured': False
        })
    
    try:
        logger.info(f"Testing SMTP connection to {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']}")
        
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=15)
        server.starttls()
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        server.quit()
        
        logger.info("SMTP connection test successful")
        return jsonify({
            'success': True,
            'message': 'SMTP connection successful',
            'smtp_configured': True,
            'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
            'smtp_port': EMAIL_CONFIG['SMTP_PORT']
        })
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication Failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'smtp_configured': True
        })
        
    except Exception as e:
        error_msg = f"SMTP Connection Failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'smtp_configured': True
        })

@app.route('/export/pdf')
def export_pdf():
    """Export traffic data as PDF"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        # Get data
        history = get_traffic_history(100)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        elements.append(Paragraph("Traffic Management System Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(timestamp)
        elements.append(Spacer(1, 0.25*inch))
        
        # Create table data
        table_data = [['Time', 'Vehicles', 'Density', 'Signal', 'Cars', 'Mcycles', 'Buses', 'Trucks']]
        
        for item in history[:50]:
            details = item.get('vehicle_details', {})
            table_data.append([
                datetime.fromisoformat(item['timestamp']).strftime('%H:%M:%S'),
                str(item.get('vehicle_count', 0)),
                item.get('density', 'LOW'),
                item.get('signal_state', 'RED'),
                str(details.get('car', 0)),
                str(details.get('motorcycle', 0)),
                str(details.get('bus', 0)),
                str(details.get('truck', 0))
            ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'traffic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/csv')
def export_csv():
    """Export traffic data as CSV"""
    try:
        history = get_traffic_history(1000)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Total Vehicles', 'Density', 'Signal State', 
                        'Cars', 'Motorcycles', 'Buses', 'Trucks', 'Control Mode'])
        
        # Write data
        for item in history:
            details = item.get('vehicle_details', {})
            writer.writerow([
                item['timestamp'],
                item.get('vehicle_count', 0),
                item.get('density', 'LOW'),
                item.get('signal_state', 'RED'),
                details.get('car', 0),
                details.get('motorcycle', 0),
                details.get('bus', 0),
                details.get('truck', 0),
                item.get('control_mode', 'AUTO')
            ])
        
        # Prepare response
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )
        
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/json')
def export_json():
    """Export traffic data as JSON"""
    try:
        history = get_traffic_history(1000)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        return Response(
            json.dumps(history, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename=traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
        )
        
    except Exception as e:
        logger.error(f"JSON export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signal_status')
def signal_status():
    """Get current signal controller status (for debugging)"""
    with traffic_state.lock:
        remaining = max(0, traffic_state.signal_timer - (time.time() - traffic_state.last_switch_time))
        return jsonify({
            'signal_state': traffic_state.signal_state,
            'base_duration': traffic_state.base_duration,
            'remaining_time': round(remaining, 1),
            'last_switch': traffic_state.last_switch_time,
            'density_level': traffic_state.density_level,
            'control_mode': traffic_state.control_mode,
            'red_durations': {
                'low': Config.SIGNAL_DURATIONS['RED']['LOW'],
                'medium': Config.SIGNAL_DURATIONS['RED']['MEDIUM'],
                'high': Config.SIGNAL_DURATIONS['RED']['HIGH']
            },
            'green_durations': {
                'low': Config.SIGNAL_DURATIONS['GREEN']['LOW'],
                'medium': Config.SIGNAL_DURATIONS['GREEN']['MEDIUM'],
                'high': Config.SIGNAL_DURATIONS['GREEN']['HIGH']
            },
            'yellow': Config.SIGNAL_DURATIONS['YELLOW']
        })

# ==================== Main Entry Point ====================

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load models
    load_models()
    
    # Start the continuous signal controller thread
    start_signal_controller()
    
    # Try to load saved prediction model
    model_loaded = load_saved_model()
    
    if model_loaded:
        logger.info("Using saved prediction model from disk")
        # Generate predictions using loaded model
        from ml_model import generate_predictions
        traffic_state.current_predictions = generate_predictions()
    else:
        logger.info("No saved model found, using demo predictions")
        # Generate initial demo predictions
        from ml_model import generate_demo_predictions
        traffic_state.current_predictions = generate_demo_predictions()
    
    # Analyze hourly patterns
    from ml_model import analyze_hourly_patterns
    analyze_hourly_patterns()
    
    # Log startup
    from database_manager import log_system_event
    log_system_event("INFO", "System started")
    
    # Log email configuration on startup
    logger.info(f"Email configuration on startup: Enabled={EMAIL_CONFIG['ALERT_ENABLED']}, Email='{EMAIL_CONFIG['ALERT_EMAIL']}'")
    logger.info(f"SMTP configured: {bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])}")
    
    # Register cleanup function
    from traffic_core import cleanup
    atexit.register(cleanup)
    
    # Run Flask app with single thread to avoid duplicate signal controllers
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,  # Disable reloader to prevent duplicate threads
        threaded=True
    )