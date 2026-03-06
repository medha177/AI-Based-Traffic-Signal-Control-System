"""
AI Smart Traffic Management System - Database Manager Module
Contains database initialization, SQLite connection logic, and all database read/write functions
"""

import sqlite3
import json
import logging
from datetime import datetime
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
LOCAL_DB_PATH = 'traffic_data.db'

def init_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    # Create traffic history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_count INTEGER DEFAULT 0,
            density TEXT DEFAULT 'LOW',
            signal_state TEXT DEFAULT 'RED',
            mode TEXT DEFAULT 'STANDBY',
            control_mode TEXT DEFAULT 'AUTO',
            fps REAL DEFAULT 0,
            session_id TEXT,
            vehicle_details TEXT DEFAULT '{}',
            hour_of_day INTEGER,
            day_of_week INTEGER
        )
    ''')
    
    # Create index on timestamp for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON traffic_history(timestamp DESC)
    ''')
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time TEXT,
            end_time TEXT,
            status TEXT DEFAULT 'active',
            control_mode TEXT DEFAULT 'AUTO',
            total_records INTEGER DEFAULT 0
        )
    ''')
    
    # Create mode changes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mode_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            old_mode TEXT,
            new_mode TEXT,
            session_id TEXT
        )
    ''')
    
    # Create manual overrides table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS manual_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            old_state TEXT,
            new_state TEXT,
            session_id TEXT
        )
    ''')
    
    # Create system_events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            description TEXT,
            session_id TEXT
        )
    ''')
    
    # Create settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            updated_at TEXT
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hour INTEGER,
            day_of_week INTEGER,
            predicted_density TEXT,
            predicted_count REAL,
            confidence REAL,
            model_used TEXT,
            training_accuracy REAL
        )
    ''')
    
    # Create model_metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_time TEXT,
            samples_used INTEGER,
            accuracy REAL,
            model_type TEXT,
            features TEXT
        )
    ''')
    
    # Create email_settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_email TEXT,
            alert_enabled INTEGER DEFAULT 0,
            updated_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")
    
    # Import traffic_state and initialize session
    try:
        from traffic_core import traffic_state
        traffic_state.init_session()
    except:
        pass

def store_traffic_data(force=False):
    """Store current traffic data in SQLite database"""
    # Import here to avoid circular imports
    from traffic_core import traffic_state
    
    try:
        current_time = datetime.now()
        hour = current_time.hour
        day = current_time.weekday()
        
        # Convert vehicle counts to JSON for storage
        vehicle_details = json.dumps(dict(traffic_state.vehicle_counts))
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_history 
            (timestamp, vehicle_count, density, signal_state, mode, control_mode, fps, 
             session_id, vehicle_details, hour_of_day, day_of_week)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_time.isoformat(),
            traffic_state.total_vehicles,
            traffic_state.density_level,
            traffic_state.signal_state,
            traffic_state.current_mode,
            traffic_state.control_mode,
            traffic_state.fps,
            traffic_state.session_id,
            vehicle_details,
            hour,
            day
        ))
        
        # Update session record count
        cursor.execute('''
            UPDATE sessions 
            SET total_records = total_records + 1 
            WHERE session_id = ?
        ''', (traffic_state.session_id,))
        
        conn.commit()
        conn.close()
        
        if force:
            logger.info(f"FORCED: Traffic data stored - {traffic_state.total_vehicles} vehicles at {current_time.strftime('%H:%M:%S')}")
        
        return True
    except Exception as e:
        logger.error(f"Error storing data: {e}")
        log_system_event("ERROR", f"Data storage failed: {e}")
        return False

def get_traffic_history(limit=100, session_id=None):
    """Get traffic history from database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute(
                "SELECT * FROM traffic_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM traffic_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts and parse vehicle_details
        history = []
        for row in rows:
            item = dict(row)
            try:
                item['vehicle_details'] = json.loads(item['vehicle_details'])
            except:
                item['vehicle_details'] = {}
            history.append(item)
            
        logger.info(f"Retrieved {len(history)} history records")
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return []

def get_traffic_analytics():
    """Get traffic analytics from database with graph data"""
    try:
        history = get_traffic_history(1000)  # Get last 1000 records for better analytics
        
        if not history:
            logger.info("No history data for analytics")
            return {
                'average_vehicles': 0,
                'max_vehicles': 0,
                'min_vehicles': 0,
                'density_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
                'control_mode_distribution': {'AUTO': 0, 'MANUAL': 0},
                'vehicle_type_distribution': {},
                'hourly_statistics': {},
                'peak_hours': [],
                'total_data_points': 0,
                'start_time': None,
                'end_time': None,
                'timeline_data': [],
                'density_over_time': []
            }
        
        # Calculate analytics
        vehicle_counts = [item.get('vehicle_count', 0) for item in history]
        densities = [item.get('density', 'LOW') for item in history]
        control_modes = [item.get('control_mode', 'AUTO') for item in history]
        
        # Aggregate vehicle types
        vehicle_types = defaultdict(int)
        for item in history:
            details = item.get('vehicle_details', {})
            for vtype, count in details.items():
                vehicle_types[vtype] += count
        
        # Hourly averages and timeline data
        hourly_avg = defaultdict(list)
        timeline_data = []
        density_over_time = []
        
        # Sort history by timestamp for proper timeline
        history_sorted = sorted(history, key=lambda x: x.get('timestamp', ''))
        
        for item in history_sorted[-100:]:  # Last 100 records for timeline
            try:
                timestamp = datetime.fromisoformat(item['timestamp'])
                hour = timestamp.hour
                count = item.get('vehicle_count', 0)
                density = item.get('density', 'LOW')
                
                hourly_avg[hour].append(count)
                
                # Add to timeline data
                timeline_data.append({
                    'time': timestamp.strftime('%H:%M:%S'),
                    'count': count,
                    'density': density
                })
                
                # Add density over time for graph
                density_over_time.append({
                    'time': timestamp.strftime('%H:%M'),
                    'density': density,
                    'count': count
                })
            except Exception as e:
                logger.error(f"Error processing timestamp: {e}")
                continue
        
        # Create hourly statistics with all 24 hours
        hourly_stats = {}
        for hour in range(24):
            counts = hourly_avg.get(hour, [0])
            if counts and counts != [0]:
                avg_count = sum(counts) / len(counts)
                max_count = max(counts)
                min_count = min(counts)
            else:
                avg_count = 0
                max_count = 0
                min_count = 0
            
            hourly_stats[str(hour)] = {  # Use string keys for JSON
                'avg': round(avg_count, 1),
                'max': max_count,
                'min': min_count,
                'samples': len(counts) if counts != [0] else 0
            }
        
        # Get peak hours (top 5)
        peak_hours_data = []
        for hour, stats in hourly_stats.items():
            if stats['avg'] > 0:
                peak_hours_data.append({
                    'hour': int(hour),
                    'avg_count': stats['avg']
                })
        peak_hours_data.sort(key=lambda x: x['avg_count'], reverse=True)
        peak_hours = peak_hours_data[:5] if peak_hours_data else []
        
        # Calculate density distribution percentages
        total_records = len(history)
        density_dist = {
            'LOW': densities.count('LOW'),
            'MEDIUM': densities.count('MEDIUM'),
            'HIGH': densities.count('HIGH')
        }
        
        # Add percentages
        density_percentages = {}
        for level, count in density_dist.items():
            density_percentages[level] = {
                'count': count,
                'percentage': round((count / total_records * 100), 1) if total_records > 0 else 0
            }
        
        analytics = {
            'average_vehicles': round(sum(vehicle_counts) / len(vehicle_counts), 2) if vehicle_counts else 0,
            'max_vehicles': max(vehicle_counts) if vehicle_counts else 0,
            'min_vehicles': min(vehicle_counts) if vehicle_counts else 0,
            'density_distribution': density_dist,
            'density_percentages': density_percentages,
            'control_mode_distribution': {
                'AUTO': control_modes.count('AUTO'),
                'MANUAL': control_modes.count('MANUAL')
            },
            'vehicle_type_distribution': dict(vehicle_types),
            'hourly_statistics': hourly_stats,
            'peak_hours': peak_hours,
            'total_data_points': len(history),
            'start_time': history[-1].get('timestamp') if history else None,
            'end_time': history[0].get('timestamp') if history else None,
            'timeline_data': timeline_data[-50:],  # Last 50 points
            'density_over_time': density_over_time[-30:],  # Last 30 points for density graph
            'recent_counts': vehicle_counts[-20:] if vehicle_counts else []  # Last 20 counts for trend
        }
        
        logger.info(f"Analytics calculated from {len(history)} records")
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return {
            'average_vehicles': 0,
            'max_vehicles': 0,
            'min_vehicles': 0,
            'density_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
            'density_percentages': {'LOW': {'count': 0, 'percentage': 0}, 
                                    'MEDIUM': {'count': 0, 'percentage': 0}, 
                                    'HIGH': {'count': 0, 'percentage': 0}},
            'control_mode_distribution': {'AUTO': 0, 'MANUAL': 0},
            'vehicle_type_distribution': {},
            'hourly_statistics': {},
            'peak_hours': [],
            'total_data_points': 0,
            'start_time': None,
            'end_time': None,
            'timeline_data': [],
            'density_over_time': [],
            'recent_counts': []
        }

def get_timeline_data(limit=50):
    """Get timeline data specifically for graphs"""
    try:
        history = get_traffic_history(limit)
        
        timeline = []
        for item in history:
            try:
                timestamp = datetime.fromisoformat(item['timestamp'])
                timeline.append({
                    'time': timestamp.strftime('%H:%M:%S'),
                    'count': item.get('vehicle_count', 0),
                    'density': item.get('density', 'LOW'),
                    'signal': item.get('signal_state', 'RED')
                })
            except:
                continue
        
        # Sort by time
        timeline.sort(key=lambda x: x['time'])
        
        return timeline
    except Exception as e:
        logger.error(f"Error getting timeline data: {e}")
        return []

def log_system_event(event_type, description):
    """Log system event to database"""
    # Import here to avoid circular imports
    try:
        from traffic_core import traffic_state
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO system_events (timestamp, event_type, description, session_id) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), event_type, description, traffic_state.session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log event: {e}")

def log_mode_change(old_mode, new_mode):
    """Log mode change to database"""
    # Import here to avoid circular imports
    try:
        from traffic_core import traffic_state
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO mode_changes (timestamp, old_mode, new_mode, session_id) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), old_mode, new_mode, traffic_state.session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to store mode change: {e}")

def log_manual_override(old_state, new_state):
    """Log manual override to database"""
    # Import here to avoid circular imports
    try:
        from traffic_core import traffic_state
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO manual_overrides (timestamp, old_state, new_state, session_id) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), old_state, new_state, traffic_state.session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to store manual override: {e}")

def update_session_record_count(session_id):
    """Update session record count"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET total_records = total_records + 1 WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update session record count: {e}")

def complete_session(session_id):
    """Mark session as completed"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET end_time = ?, status = ? WHERE session_id = ?",
            (datetime.now().isoformat(), 'completed', session_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Session {session_id} completed")
    except Exception as e:
        logger.error(f"Failed to update session: {e}")

def save_prediction_to_db(hour, day, density, count, confidence, model_used, accuracy):
    """Save a prediction to the database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, hour, day_of_week, predicted_density, predicted_count, confidence, model_used, training_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            hour,
            day,
            density,
            round(count, 1),
            confidence,
            model_used,
            accuracy
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")

def save_model_metadata(training_time, samples_used, accuracy, model_type, features):
    """Save model metadata to database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_metadata 
            (training_time, samples_used, accuracy, model_type, features)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            training_time,
            samples_used,
            accuracy,
            model_type,
            json.dumps(features)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save model metadata: {e}")

def get_latest_model_metadata():
    """Get latest model metadata from database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT accuracy, samples_used, training_time 
            FROM model_metadata 
            ORDER BY training_time DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        return None

def get_recent_predictions(limit=24):
    """Get most recent predictions from database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p1.* 
            FROM predictions p1
            INNER JOIN (
                SELECT hour, MAX(timestamp) as max_timestamp
                FROM predictions
                GROUP BY hour
            ) p2 ON p1.hour = p2.hour AND p1.timestamp = p2.max_timestamp
            ORDER BY p1.hour ASC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            pred = dict(row)
            predictions.append(pred)
        
        return predictions
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        return []

def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        stats = {}
        
        # Get table sizes
        cursor.execute("SELECT COUNT(*) FROM traffic_history")
        stats['traffic_history'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        stats['sessions'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mode_changes")
        stats['mode_changes'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM manual_overrides")
        stats['manual_overrides'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM system_events")
        stats['system_events'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['predictions'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_metadata")
        stats['model_metadata'] = cursor.fetchone()[0]
        
        # Get latest record
        cursor.execute("SELECT timestamp, vehicle_count FROM traffic_history ORDER BY timestamp DESC LIMIT 1")
        latest = cursor.fetchone()
        if latest:
            stats['latest_record'] = {
                'timestamp': latest[0],
                'vehicles': latest[1]
            }
        
        # Get database file size
        if os.path.exists(LOCAL_DB_PATH):
            stats['database_size_kb'] = round(os.path.getsize(LOCAL_DB_PATH) / 1024, 2)
        
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}