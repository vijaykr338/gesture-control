import cv2
import numpy as np
import time
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from config_manager import config_manager
from event_system import event_bus, GestureEvent
from detection_models import model_manager

# Import all functions from hand_landmark module
from hand_landmark import *

@dataclass
class ProcessingResult:
    """Results from gesture processing"""
    regions: List[Any]
    detected_gestures: List[str]
    hand_count: int
    processing_time: float
    frame_info: Dict[str, Any]

# Add all the missing functions from your notebook
def calculate_angle(p1, p2, p3):
    """Calculate angle at point p2 formed by p1-p2-p3"""
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 180
    
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def detect_index_finger_bend(landmarks):
    """Detect if index finger is bent using angle calculation"""
    try:
        mcp = landmarks[5]  # Base joint
        pip = landmarks[6]  # Middle joint (where we measure angle)
        dip = landmarks[8]  # Top joint
        
        angle = calculate_angle(mcp, pip, dip)
        return angle
        
    except Exception as e:
        print(f"Error in finger bend detection: {e}")
        return 180

def detect_middle_finger_bend(landmarks):
    """Detect if middle finger is bent using angle calculation"""
    try:
        mcp = landmarks[9]   # Middle finger MCP (base joint)
        pip = landmarks[10]  # Middle finger PIP (middle joint) 
        dip = landmarks[12]  # Middle finger DIP (NOT tip - for consistent measurement)
        
        angle = calculate_angle(mcp, pip, dip)
        return angle
        
    except Exception as e:
        print(f"Error in middle finger bend detection: {e}")
        return 180

def detect_finger_perpendicularity(landmarks):
    """Detect angle between index and middle finger directions"""
    try:
        # Index finger direction vector (from MCP to PIP)
        index_mcp = landmarks[5]   # Index MCP (base)
        index_pip = landmarks[6]   # Index PIP (middle joint)
        
        # Middle finger direction vector (from MCP to PIP)  
        middle_mcp = landmarks[9]  # Middle MCP (base)
        middle_pip = landmarks[10] # Middle PIP (middle joint)
        
        # Calculate direction vectors
        index_vector = [index_pip[0] - index_mcp[0], index_pip[1] - index_mcp[1]]
        middle_vector = [middle_pip[0] - middle_mcp[0], middle_pip[1] - middle_mcp[1]]
        
        # Calculate angle between vectors using dot product
        dot_product = index_vector[0] * middle_vector[0] + index_vector[1] * middle_vector[1]
        
        # Calculate magnitudes
        index_mag = math.sqrt(index_vector[0]**2 + index_vector[1]**2)
        middle_mag = math.sqrt(middle_vector[0]**2 + middle_vector[1]**2)
        
        if index_mag == 0 or middle_mag == 0:
            return 90  # Default to perpendicular if calculation fails
        
        # Calculate angle between finger directions
        cos_angle = dot_product / (index_mag * middle_mag)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle_between = math.degrees(math.acos(abs(cos_angle)))
        
        return angle_between
        
    except Exception as e:
        print(f"Error in perpendicularity detection: {e}")
        return 90  # Default to perpendicular

def are_fingers_parallel(landmarks, parallel_threshold=30):
    """Check if fingers are parallel (can do both-finger gestures)"""
    angle_between = detect_finger_perpendicularity(landmarks)
    return angle_between < parallel_threshold

def are_fingers_perpendicular(landmarks, perpendicular_threshold=20):
    """Check if fingers are perpendicular (only index gesture)"""
    angle_between = detect_finger_perpendicularity(landmarks)
    return abs(angle_between - 90) < perpendicular_threshold

def detect_gesture_type(landmarks, bend_threshold):
    """Detect gesture type based on finger angles and relationships"""
    try:
        # Get individual finger angles
        index_angle = detect_index_finger_bend(landmarks)
        middle_angle = detect_middle_finger_bend(landmarks)
        
        # Check finger relationship
        fingers_parallel = are_fingers_parallel(landmarks)
        fingers_perpendicular = are_fingers_perpendicular(landmarks)
        finger_angle_between = detect_finger_perpendicularity(landmarks)
        
        # Define finger bend states
        def get_finger_state(angle):
            if angle < 60:
                return "RELAXED"      # Very curled (natural resting)
            elif angle < 130:
                return "BENT"         # Intentionally bent for gesture
            else:
                return "EXTENDED"     # Straight finger
        
        index_state = get_finger_state(index_angle)
        middle_state = get_finger_state(middle_angle)
        
        # Smart gesture logic
        gesture_type = "none"
        
        if index_state == "BENT":
            if fingers_parallel and middle_state == "BENT":
                gesture_type = "index_middle_both"
            elif fingers_perpendicular:
                gesture_type = "index_only"
            elif not fingers_parallel and middle_state in ["EXTENDED", "RELAXED"]:
                gesture_type = "index_only"
        
        return {
            'index_angle': index_angle,
            'middle_angle': middle_angle,
            'index_state': index_state,
            'middle_state': middle_state,
            'gesture_type': gesture_type,
            'finger_angle_between': finger_angle_between,
            'fingers_parallel': fingers_parallel,
            'fingers_perpendicular': fingers_perpendicular
        }
        
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return {
            'index_angle': 180,
            'middle_angle': 180,
            'index_state': "EXTENDED",
            'middle_state': "EXTENDED", 
            'gesture_type': "none",
            'finger_angle_between': 90,
            'fingers_parallel': False,
            'fingers_perpendicular': True
        }

def process_finger_detection(region, params):
    """Process finger detection with gesture mapping system"""
    if not (hasattr(region, 'landmarks') and params['enable_finger_detection']):
        return
    
    try:
        # Gesture detection for compatibility/debugging
        gesture_data = detect_gesture_type(region.landmarks, params['bend_angle_threshold'])
        hand_type = "right" if region.handedness > 0.5 else "left"
        
        # Store results
        region.index_angle = gesture_data['index_angle']
        region.middle_angle = gesture_data['middle_angle']
        region.index_state = gesture_data['index_state']
        region.middle_state = gesture_data['middle_state']
        region.gesture_type = gesture_data['gesture_type']
        region.hand_type = hand_type
        
        # Store perpendicular data for rendering
        region.finger_angle_between = gesture_data['finger_angle_between']
        region.fingers_parallel = gesture_data['fingers_parallel']
        region.fingers_perpendicular = gesture_data['fingers_perpendicular']
        
    except Exception as e:
        print(f"Error processing finger detection: {e}")

def process_application_modes(region, params, app_modes):
    """Simplified application modes processing for Phase 2"""
    if not hasattr(region, 'landmarks'):
        return
    
    current_mode = app_modes['current_mode']
    if current_mode == 'disabled' or current_mode not in app_modes:
        return
    
    # Basic gesture detection and execution would go here
    # For now, just mark that we processed the region
    region.processed_for_app_mode = True

class GestureProcessor:
    """Handles the core gesture processing logic"""
    
    def __init__(self):
        self.previous_regions = []
        self.state_data = {}
        self.performance_stats = {
            'frame_count': 0,
            'average_fps': 0,
            'last_fps_update': time.time()
        }
    
    def process_frame(self, frame: np.ndarray) -> ProcessingResult:
        """Process a single frame for gesture detection"""
        start_time = time.time()
        
        # Get current configuration
        config = config_manager.get_legacy_params_dict()
        
        # Preprocess frame
        frame_h, frame_w = frame.shape[:2]
        input_size = config['input_size']
        resized_frame = cv2.resize(frame, (input_size, input_size))
        
        # Determine if palm detection is needed
        need_palm_detection = self._should_run_palm_detection(config)
        
        # Process regions
        if need_palm_detection:
            regions = self._run_palm_detection(resized_frame, config)
        else:
            regions = self.previous_regions.copy()
        
        # Process landmarks and gestures
        processed_regions = self._process_landmarks_and_gestures(regions, resized_frame, config)
        
        # Process application modes
        detected_gestures = self._process_application_modes(processed_regions, config)
        
        # Update state
        self.previous_regions = processed_regions.copy()
        self._update_performance_stats()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = ProcessingResult(
            regions=processed_regions,
            detected_gestures=detected_gestures,
            hand_count=len(processed_regions),
            processing_time=processing_time,
            frame_info={
                'width': frame_w,
                'height': frame_h,
                'input_size': input_size,
                'palm_detection_used': need_palm_detection
            }
        )
        
        # Publish events
        self._publish_events(result)
        
        return result
    
    def _should_run_palm_detection(self, config: dict) -> bool:
        """Determine if palm detection should run"""
        # Use smart palm detection state machine from your existing code
        current_hand_count = len(self.previous_regions)
        
        # Apply state machine logic (simplified version)
        if config['always_run_palm_detection']:
            return True
        
        if not self.previous_regions:
            return True
        
        # Check landmark scores
        for region in self.previous_regions:
            if not hasattr(region, 'lm_score') or region.lm_score < config['landmark_score_for_palm_redetection_threshold']:
                return True
        
        return False
    
    def _run_palm_detection(self, resized_frame: np.ndarray, config: dict) -> List[Any]:
        """Run palm detection on frame"""
        try:
            compiled_model = model_manager.get_compiled_model('palm_detection')
            if not compiled_model:
                return []
            
            # Generate anchors
            anchors2 = generate_anchors(options)
            anchors2_np = np.array(anchors2)
            
            # Run detection (using your existing function)
            input_tensor = np.expand_dims(resized_frame, axis=0)
            results = compiled_model([input_tensor])
            
            # Process results (simplified - you can use your existing decode_bboxes function)
            output_node_regressors = compiled_model.output("Identity")
            output_node_scores = compiled_model.output("Identity_1")
            
            regressors_tensor = results[output_node_regressors]
            scores_tensor = results[output_node_scores]
            
            raw_scores = scores_tensor[0, :, 0]
            raw_bboxes_and_keypoints = regressors_tensor[0]
            
            regions = decode_bboxes(
                config['score_threshold'], raw_scores, raw_bboxes_and_keypoints, anchors2_np
            )
            
            if regions:
                regions = non_max_suppression(regions, config['nms_threshold'])
                detections_to_rect(regions)
                rect_transformation(regions, config['input_size'], config['input_size'])
            
            return regions
            
        except Exception as e:
            print(f"Error in palm detection: {e}")
            return []
    
    def _process_landmarks_and_gestures(self, regions: List[Any], resized_frame: np.ndarray, config: dict) -> List[Any]:
        """Process landmarks and gestures for regions"""
        if not regions or not config['show_landmarks']:
            return []
        
        processed_regions = []
        
        # Get compiled models
        landmark_model = model_manager.get_compiled_model('hand_landmarks')
        gesture_model = model_manager.get_compiled_model('gesture_embedder')
        classifier_model = model_manager.get_compiled_model('gesture_classifier')
        
        if not landmark_model:
            return []
        
        for region_idx, region in enumerate(regions):
            if not hasattr(region, 'rect_points'):
                continue
            
            try:
                # Process landmarks (using your existing functions)
                hand_crop_bgr = warp_rect_img(region.rect_points, resized_frame, 224, 224)
                hand_crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
                hand_input = np.expand_dims(hand_crop_rgb, axis=0).astype(np.float32) / 255.0
                
                lm_results = landmark_model([hand_input])
                
                if config['show_static_gestures'] and gesture_model and classifier_model:
                    lm_postprocess_with_gesture_classification(
                        region, lm_results, self.previous_regions,
                        gesture_model, classifier_model,
                        alpha=config['smoothing_alpha'],
                        iou_threshold=config['iou_match_threshold']
                    )
                else:
                    lm_postprocess(region, lm_results, self.previous_regions,
                                 alpha=config['smoothing_alpha'],
                                 iou_threshold=config['iou_match_threshold'])
                
                # Process finger detection
                process_finger_detection(region, config)
                
                processed_regions.append(region)
                
            except Exception as e:
                print(f"Error processing region {region_idx}: {e}")
        
        return processed_regions
    
    def _process_application_modes(self, processed_regions: List[Any], config: dict) -> List[str]:
        """Process application modes and return detected gestures"""
        detected_gestures = []
        
        for region in processed_regions:
            try:
                # Process application modes (using your existing function)
                process_application_modes(region, config, config['app_modes'])
                
                # Collect detected gestures
                if hasattr(region, 'detected_gesture_ids'):
                    detected_gestures.extend(region.detected_gesture_ids)
                
                if hasattr(region, 'active_gesture'):
                    detected_gestures.append(region.active_gesture)
                    
            except Exception as e:
                print(f"Error in application mode processing: {e}")
        
        return detected_gestures
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        self.performance_stats['frame_count'] += 1
        current_time = time.time()
        
        if current_time - self.performance_stats['last_fps_update'] >= 1.0:
            # Update FPS every second
            elapsed = current_time - self.performance_stats['last_fps_update']
            fps = self.performance_stats['frame_count'] / elapsed
            self.performance_stats['average_fps'] = fps
            self.performance_stats['frame_count'] = 0
            self.performance_stats['last_fps_update'] = current_time
    
    def _publish_events(self, result: ProcessingResult):
        """Publish events based on processing results"""
        current_time = datetime.now()
        
        # Hand count changes
        previous_hand_count = getattr(self, '_previous_hand_count', 0)
        if result.hand_count != previous_hand_count:
            event_bus.publish(GestureEvent(
                event_type='hand_count_changed',
                timestamp=current_time,
                data={
                    'previous_count': previous_hand_count,
                    'current_count': result.hand_count,
                    'regions': result.regions
                }
            ))
            self._previous_hand_count = result.hand_count
        
        # Gesture detection events
        for gesture in result.detected_gestures:
            event_bus.publish(GestureEvent(
                event_type='gesture_detected',
                timestamp=current_time,
                data={
                    'gesture_name': gesture,
                    'hand_count': result.hand_count
                }
            ))
        
        # Performance update
        if self.performance_stats['frame_count'] % 30 == 0:  # Every 30 frames
            event_bus.publish(GestureEvent(
                event_type='performance_update',
                timestamp=current_time,
                data={
                    'fps': self.performance_stats['average_fps'],
                    'processing_time': result.processing_time,
                    'hand_count': result.hand_count
                }
            ))
    
    def reset(self):
        """Reset processor state"""
        self.previous_regions = []
        self.state_data = {}
        self.performance_stats = {
            'frame_count': 0,
            'average_fps': 0,
            'last_fps_update': time.time()
        }

# Global processor instance
gesture_processor = GestureProcessor()