"""
AI Smart Traffic Management System - Machine Learning Module
Contains prediction model training, RandomForest model logic, model saving/loading,
hourly pattern analysis, and prediction generation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import logging
import sqlite3
import json
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_prediction_model.pkl')

def load_saved_model():
    """Load saved prediction model from disk if it exists"""
    # Import here to avoid circular imports
    from traffic_core import traffic_state
    
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Found saved model at {MODEL_PATH}, loading...")
            with open(MODEL_PATH, 'rb') as f:
                traffic_state.prediction_model = pickle.load(f)
            
            # Load model metadata from database
            from database_manager import get_latest_model_metadata
            result = get_latest_model_metadata()
            
            if result:
                traffic_state.model_accuracy = result[0]
                traffic_state.model_samples = result[1]
                traffic_state.last_training_time = datetime.fromisoformat(result[2]).timestamp()
                logger.info(f"Loaded model metadata - Accuracy: {result[0]:.1f}%, Samples: {result[1]}")
            else:
                logger.info("No metadata found for loaded model")
            
            logger.info("Saved prediction model loaded successfully")
            traffic_state.log_event("INFO", "Saved prediction model loaded from disk")
            return True
        else:
            logger.info(f"No saved model found at {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Failed to load saved model: {e}")
        traffic_state.log_event("ERROR", f"Failed to load saved model: {e}")
        return False

def save_model_to_disk():
    """Save the trained prediction model to disk"""
    # Import here to avoid circular imports
    from traffic_core import traffic_state
    
    if traffic_state.prediction_model is None:
        logger.warning("No model to save")
        return False
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(traffic_state.prediction_model, f)
        
        logger.info(f"Prediction model saved to {MODEL_PATH}")
        traffic_state.log_event("INFO", f"Model saved to disk with accuracy {traffic_state.model_accuracy:.1f}%")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to disk: {e}")
        traffic_state.log_event("ERROR", f"Failed to save model: {e}")
        return False

def analyze_hourly_patterns():
    """Analyze hourly traffic patterns from historical data"""
    # Import here to avoid circular imports
    from traffic_core import traffic_state
    from database_manager import LOCAL_DB_PATH
    
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        query = """
            SELECT hour_of_day, AVG(vehicle_count) as avg_count, 
                   MAX(vehicle_count) as max_count,
                   COUNT(*) as sample_count
            FROM traffic_history 
            WHERE vehicle_count > 0
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return {}
        
        # Calculate peak hours (top 3 hours with highest average)
        peak_hours = df.nlargest(3, 'avg_count')[['hour_of_day', 'avg_count']].to_dict('records')
        
        # Create hourly analysis
        hourly_analysis = {}
        for _, row in df.iterrows():
            hour = int(row['hour_of_day'])
            avg = row['avg_count']
            
            if avg <= 7:
                density = "LOW"
            elif avg <= 15:
                density = "MEDIUM"
            else:
                density = "HIGH"
            
            hourly_analysis[hour] = {
                'hour': hour,
                'avg_count': round(avg, 1),
                'max_count': int(row['max_count']),
                'sample_count': int(row['sample_count']),
                'density': density
            }
        
        traffic_state.hourly_analysis = hourly_analysis
        traffic_state.peak_hours = peak_hours
        
        logger.info(f"Hourly analysis complete. Peak hours: {peak_hours}")
        return hourly_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing hourly patterns: {e}")
        return {}

def generate_predictions():
    """Generate density predictions for next 24 hours"""
    # Import here to avoid circular imports
    from traffic_core import traffic_state, Config, EMAIL_CONFIG, send_alert
    from database_manager import LOCAL_DB_PATH, save_prediction_to_db
    
    if traffic_state.prediction_model is None:
        logger.info("No trained model available for predictions")
        return generate_demo_predictions()
    
    try:
        predictions = []
        current_time = datetime.now()
        
        # Get average vehicle count for reference
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(vehicle_count) FROM traffic_history WHERE vehicle_count > 0 LIMIT 100")
        result = cursor.fetchone()
        avg_count = result[0] if result and result[0] else 10
        conn.close()
        
        for hour_offset in range(1, 25):
            pred_time = current_time + timedelta(hours=hour_offset)
            hour = pred_time.hour
            day = pred_time.weekday()
            
            # Create features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            # Get historical average for this hour as prev_hour
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(vehicle_count) FROM traffic_history WHERE hour_of_day = ? LIMIT 50",
                (max(0, hour-1),)
            )
            result = cursor.fetchone()
            prev_hour_avg = result[0] if result and result[0] else avg_count
            conn.close()
            
            features = np.array([[hour_sin, hour_cos, day_sin, day_cos, prev_hour_avg]])
            
            # Predict
            try:
                pred_count = traffic_state.prediction_model.predict(features)[0]
            except:
                pred_count = avg_count
            
            # Ensure positive count
            pred_count = max(1, pred_count)
            
            # Determine density based on predicted count
            if pred_count <= 7:
                density = "LOW"
            elif pred_count <= 15:
                density = "MEDIUM"
            else:
                density = "HIGH"
            
            prediction = {
                'hour': hour_offset,
                'time': pred_time.strftime('%H:00'),
                'hour_of_day': hour,
                'predicted_count': round(pred_count, 1),
                'density': density,
                'confidence': round(traffic_state.model_accuracy, 1)
            }
            predictions.append(prediction)
            
            # Store prediction in database
            save_prediction_to_db(
                hour_offset, day, density, pred_count,
                traffic_state.model_accuracy / 100,
                Config.PREDICTION_MODEL,
                traffic_state.model_accuracy
            )
        
        logger.info(f"Generated {len(predictions)} hourly predictions")
        
        # Store predictions in traffic_state for quick access
        traffic_state.current_predictions = predictions
        
        # Analyze hourly patterns
        analyze_hourly_patterns()
        
        # Send alert for new predictions if enabled
        if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL'] and Config.ALERT_TYPES.get('model_update', False):
            peak_hours = [p for p in predictions if p['density'] == 'HIGH'][:3]
            peak_info = "<br>".join([f"• {p['time']}: {p['density']} ({p['predicted_count']} vehicles)" 
                                     for p in peak_hours]) if peak_hours else "No peak hours predicted"
            
            message = f"""
            <h3>New Traffic Predictions Available</h3>
            <p><strong>Model accuracy:</strong> {traffic_state.model_accuracy:.1f}%</p>
            <p><strong>Samples used:</strong> {traffic_state.model_samples}</p>
            <h4>Peak Hours (next 24h):</h4>
            {peak_info}
            <p><small>View full predictions in the dashboard</small></p>
            """
            send_alert("New Traffic Predictions", message, 'model_update')
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction generation error: {e}")
        return generate_demo_predictions()

def generate_demo_predictions():
    """Generate demo predictions when no model is available"""
    predictions = []
    current_time = datetime.now()
    
    for hour_offset in range(1, 25):
        pred_time = current_time + timedelta(hours=hour_offset)
        hour = pred_time.hour
        
        # Generate realistic demo predictions based on time of day
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            count = 18 + (hour_offset % 5)
            density = "HIGH"
        elif 10 <= hour <= 16:  # Mid-day
            count = 12 + (hour_offset % 4)
            density = "MEDIUM"
        else:  # Night/early morning
            count = 5 + (hour_offset % 3)
            density = "LOW"
        
        predictions.append({
            'hour': hour_offset,
            'time': pred_time.strftime('%H:00'),
            'hour_of_day': hour,
            'predicted_count': count,
            'density': density,
            'confidence': 85.0
        })
    
    return predictions

def train_prediction_model():
    """
    Background thread for training prediction model
    Runs without blocking real-time detection
    """
    # Import here to avoid circular imports
    from traffic_core import traffic_state, Config, EMAIL_CONFIG, send_alert
    from database_manager import LOCAL_DB_PATH, save_model_metadata
    
    if traffic_state.training_active:
        logger.info("Training already in progress, skipping...")
        return
    
    traffic_state.training_active = True
    start_time = time.time()
    
    try:
        logger.info("Starting background model training...")
        
        # Fetch training data
        conn = sqlite3.connect(LOCAL_DB_PATH)
        query = """
            SELECT hour_of_day, day_of_week, vehicle_count, density 
            FROM traffic_history 
            WHERE vehicle_count > 0
            ORDER BY timestamp DESC 
            LIMIT 10000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < Config.MIN_TRAINING_SAMPLES:
            logger.info(f"Insufficient training samples: {len(df)} < {Config.MIN_TRAINING_SAMPLES}")
            # Generate demo predictions anyway
            traffic_state.current_predictions = generate_demo_predictions()
            traffic_state.training_active = False
            return
        
        # Prepare features
        df['density_encoded'] = df['density'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2})
        
        # Feature engineering
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Create lag features (previous hour's traffic)
        df = df.sort_values('hour_of_day')
        df['prev_hour_count'] = df['vehicle_count'].shift(1)
        df['prev_hour_count'] = df['prev_hour_count'].fillna(df['vehicle_count'].mean())
        
        features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'prev_hour_count']
        X = df[features]
        y_count = df['vehicle_count']
        
        # Split data
        X_train, X_test, y_train_count, y_test_count = train_test_split(
            X, y_count, test_size=0.2, random_state=42
        )
        
        # Train model for vehicle count prediction
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Check time limit
        if time.time() - start_time > Config.TRAINING_TIME_LIMIT:
            logger.info("Training time limit reached, stopping early")
            traffic_state.training_active = False
            return
        
        model.fit(X_train, y_train_count)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test_count, y_pred)
        r2 = r2_score(y_test_count, y_pred)
        
        # Calculate accuracy as percentage (0-100)
        y_test_mean = y_test_count.mean()
        if y_test_mean > 0:
            accuracy = max(0, min(100, (1 - mae / y_test_mean) * 100))
        else:
            accuracy = 80  # Default accuracy
        
        # Store model
        traffic_state.prediction_model = model
        traffic_state.model_accuracy = accuracy
        traffic_state.model_samples = len(df)
        traffic_state.last_training_time = time.time()
        
        # Save model to disk
        save_model_to_disk()
        
        # Store metadata
        save_model_metadata(
            datetime.now().isoformat(),
            len(df),
            accuracy,
            Config.PREDICTION_MODEL,
            features
        )
        
        logger.info(f"Model training complete - MAE: {mae:.2f}, R2: {r2:.3f}, Accuracy: {accuracy:.1f}%")
        
        # Analyze hourly patterns
        analyze_hourly_patterns()
        
        # Generate predictions for next 24 hours
        predictions = generate_predictions()
        
        # Store predictions in traffic_state
        traffic_state.current_predictions = predictions
        
        # Send alert if enabled
        if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL'] and Config.ALERT_TYPES.get('training_complete', False):
            message = f"""
            <h3>Model Training Complete</h3>
            <ul>
                <li><strong>Samples used:</strong> {len(df)}</li>
                <li><strong>Mean Absolute Error:</strong> {mae:.2f} vehicles</li>
                <li><strong>R² Score:</strong> {r2:.3f}</li>
                <li><strong>Accuracy:</strong> {accuracy:.1f}%</li>
                <li><strong>Model type:</strong> {Config.PREDICTION_MODEL}</li>
                <li><strong>Training time:</strong> {time.time() - start_time:.1f} seconds</li>
                <li><strong>Predictions generated:</strong> {len(predictions)}</li>
                <li><strong>Model saved to:</strong> {MODEL_PATH}</li>
            </ul>
            """
            send_alert("Model Training Completed", message, 'training_complete')
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        traffic_state.log_event("ERROR", f"Training failed: {e}")
        # Generate demo predictions on error
        traffic_state.current_predictions = generate_demo_predictions()
    
    finally:
        traffic_state.training_active = False