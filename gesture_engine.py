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
# BUG FIX 1: Import the missing 'process_finger_detection' function
from gesture_processor import gesture_processor, process_finger_detection
from hand_landmark import *
from application_modes import ApplicationModeManager
import pyautogui

class CompleteGestureEngine:
    """Complete gesture detection engine with full visual rendering like your notebook"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.model_manager = model_manager
        self.processor = gesture_processor
        
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        # Camera
        self.cap = None
        self.current_frame = None
        
        # Parameters for processing (legacy compatibility)
        self.params = None
        self.app_modes = None
        
        # Anchors for palm detection
        self.anchors2_np = None
        self.app_mode_manager = None
        
    def initialize(self) -> bool:
        """Initialize the complete gesture engine"""
        print("🔧 Initializing Complete Gesture Engine...")
        
        # Validate configuration
        validation = self.config_manager.validate_config()
        if validation['errors']:
            print(f"❌ Configuration errors: {validation['errors']}")
            return False
        
        # Get legacy parameters for full compatibility
        self.params = self.config_manager.get_legacy_params_dict()
        self.app_modes = self.params['app_modes']
        self.app_mode_manager = ApplicationModeManager(self.app_modes)
        self.app_mode_manager.set_engine_params(self.params)
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.0 
        
        # Initialize models
        model_paths = {
            # BUG FIX 2: Corrected the path for the palm detection model
            'palm_detection': 'mediapipeModels/hand_detector.xml',
            'hand_landmarks': 'mediapipeModels/hand_landmarks_detector.xml',
            'gesture_embedder': 'mediapipeModels/gesture_embedder.xml',
            'gesture_classifier': 'mediapipeModels/canned_gesture_classifier.xml'
        }
        
        if not self.model_manager.initialize_models(model_paths):
            print("❌ Model initialization failed!")
            return False
        
        # Initialize camera with better handling
        self.cap = None
        for camera_id in [0, 1, -1]:  # Try different camera indices
            try:
                test_cap = cv2.VideoCapture(camera_id)
                if test_cap.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera {camera_id} working!")
                        # Set optimal properties
                        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        test_cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap = test_cap
                        break
                    else:
                        test_cap.release()
                else:
                    test_cap.release()
            except Exception as e:
                print(f"Camera {camera_id} error: {e}")
                continue
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera initialization failed!")
            return False
        
        # Generate anchors
        anchors2 = generate_anchors(options)
        self.anchors2_np = np.array(anchors2)
        
        # Start event processing
        self.event_bus.start_processing()
        
        print("✅ Complete Gesture Engine initialized!")
        return True
    
    def start(self):
        """Start the engine"""
        # Re-initialize camera if it has been stopped/released
        if self.cap is None or not self.cap.isOpened():
            print("Re-initializing camera for start...")
            # This is a simplified re-init; a full re-init might be needed
            # if other resources were also released.
            self.initialize() 

        self.running = True
        self.paused = False
        if self.start_time is None:
            self.start_time = time.time()
        print("▶️ Complete Engine started!")
    
    def pause(self):
        """Pause the engine"""
        self.paused = True
        print("⏸️ Complete Engine paused!")
    
    def resume(self):
        """Resume the engine"""
        self.paused = False
        print("▶️ Complete Engine resumed!")
    
    def stop(self):
        """Stop the engine"""
        self.running = False
        self.paused = False
        if self.cap:
            self.cap.release()
            # BUG FIX 3: Set cap to None for robust restart
            self.cap = None
        self.event_bus.stop_processing()
        print("⏹️ Complete Engine stopped!")
    
    def get_frame_with_complete_processing(self):
        """Get frame with COMPLETE processing and rendering exactly like your notebook"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        original_frame = frame.copy()
        
        if not self.running or self.paused:
            # Show paused state
            cv2.putText(original_frame, "ENGINE PAUSED", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            return original_frame
        
        try:
            # Process frame EXACTLY like your notebook main loop
            frame_h, frame_w = original_frame.shape[:2]
            
            # Preprocess frame
            resized_frame_for_input = cv2.resize(frame, (self.params['input_size'], self.params['input_size']))
            
            # Smart palm detection state machine
            current_hand_count = len(self.params['previous_frame_processed_regions'])
            need_palm_detection = self._smart_palm_detection_state_machine(current_hand_count)
            
            # Fallback to original logic if state machine says no
            if not need_palm_detection:
                need_palm_detection = (
                    self.params['always_run_palm_detection'] or
                    should_run_palm_detection(
                        self.params['previous_frame_processed_regions'], 
                        self.params['landmark_score_for_palm_redetection_threshold']
                    )
                )
            
            current_regions_for_processing = []
            
            if need_palm_detection:
                # Run palm detection exactly like notebook
                regions_nms = self._run_palm_detection(resized_frame_for_input)
                
                # Apply detection smoothing exactly like notebook
                self._smooth_detection_boxes(regions_nms)
                
                current_regions_for_processing = regions_nms
                if current_regions_for_processing:
                    detections_to_rect(current_regions_for_processing)
                    rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            else:
                # Use previous frame regions exactly like notebook
                current_regions_for_processing = self.params['previous_frame_processed_regions']
                if current_regions_for_processing:
                    detections_to_rect(current_regions_for_processing)
                    rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            
            # Process landmarks and gestures exactly like notebook
            processed_regions = self._process_landmarks_and_gestures(current_regions_for_processing, resized_frame_for_input)
            
            # Process application modes for each region exactly like notebook
            for region in processed_regions:
                self._process_application_modes(region)
            
            # Execute once per frame exactly like notebook
            self._reset_hand_tracking(processed_regions)
            
            # Render results with COMPLETE visual display exactly like notebook
            self._render_results_complete(original_frame, processed_regions, frame_w, frame_h)
            
            # Display ALL status information exactly like notebook
            self._render_complete_status_info(original_frame, processed_regions, need_palm_detection)
            
            # Update previous frame regions exactly like notebook
            self.params['previous_frame_processed_regions'] = list(processed_regions)
            
            # Update performance stats
            self.frame_count += 1
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
            
            return original_frame
            
        except Exception as e:
            print(f"Error in complete frame processing: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(original_frame, f"ERROR: {str(e)[:50]}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return original_frame
    
    def _smart_palm_detection_state_machine(self, current_hand_count):
        """Smart state machine exactly like your notebook"""
        current_time = time.time()
        current_state = self.params['palm_detection_state']
        debug = self.params['state_transition_debug']
        
        # State transitions exactly like your notebook
        if current_hand_count == 0:
            if current_state != 'NO_HANDS':
                if debug: print(f"🔄 STATE: {current_state} → NO_HANDS")
                self.params['palm_detection_state'] = 'NO_HANDS'
            return True  # Always detect when no hands
            
        elif current_hand_count == 1:
            if current_state == 'NO_HANDS':
                self.params['palm_detection_state'] = 'ONE_HAND_SEARCHING'
                self.params['grace_period_start'] = current_time
                if debug: print(f"🔍 STATE: NO_HANDS → ONE_HAND_SEARCHING")
                return True
                
            elif current_state == 'ONE_HAND_SEARCHING':
                elapsed_time = current_time - self.params['grace_period_start']
                if elapsed_time >= self.params['grace_period_duration']:
                    self.params['palm_detection_state'] = 'ONE_HAND_STABLE'
                    if debug: print(f"⏰ STATE: ONE_HAND_SEARCHING → ONE_HAND_STABLE")
                    return False
                return True
                
            elif current_state == 'ONE_HAND_STABLE':
                self.params['periodic_check_counter'] += 1
                if self.params['periodic_check_counter'] >= self.params['periodic_check_interval']:
                    self.params['periodic_check_counter'] = 0
                    if debug: print(f"👀 Periodic check for 2nd hand")
                    return True
                return False
                
        elif current_hand_count >= 2:
            if current_state != 'TWO_HANDS':
                if debug: print(f"🎉 STATE: {current_state} → TWO_HANDS")
                self.params['palm_detection_state'] = 'TWO_HANDS'
            return False
        
        return False
    
    def _run_palm_detection(self, resized_frame):
        """Run palm detection exactly like your notebook"""
        try:
            compiled_model = self.model_manager.get_compiled_model('palm_detection')
            if not compiled_model:
                return []
            
            input_tensor = np.expand_dims(resized_frame, axis=0)
            
            output_node_regressors = compiled_model.output("Identity")
            output_node_scores = compiled_model.output("Identity_1")
            results = compiled_model([input_tensor])

            regressors_tensor = results[output_node_regressors]
            scores_tensor = results[output_node_scores]

            raw_scores = scores_tensor[0, :, 0]
            raw_bboxes_and_keypoints = regressors_tensor[0]

            regions_from_palm_detection = decode_bboxes(
                self.params['score_threshold'], raw_scores, raw_bboxes_and_keypoints, self.anchors2_np
            )
            
            regions_nms = []
            if regions_from_palm_detection:
                regions_nms = non_max_suppression(regions_from_palm_detection, self.params['nms_threshold'])

            return regions_nms
            
        except Exception as e:
            print(f"Error in palm detection: {e}")
            return []
    
    def _smooth_detection_boxes(self, regions_nms):
        """Apply smoothing exactly like your notebook"""
        if not (regions_nms and self.params['previous_frame_processed_regions']):
            return
        
        for current_new_region in regions_nms:
            best_match_prev = None
            max_iou_for_smoothing = 0.0
            
            for prev_reg in self.params['previous_frame_processed_regions']:
                if hasattr(prev_reg, 'pd_box'):
                    iou_val = calculate_iou(current_new_region.pd_box, prev_reg.pd_box)
                    if iou_val > max_iou_for_smoothing:
                        max_iou_for_smoothing = iou_val
                        best_match_prev = prev_reg
            
            if best_match_prev and max_iou_for_smoothing > 0.15:
                for i_coord in range(4):
                    current_new_region.pd_box[i_coord] = (
                        self.params['detection_smoothing_alpha'] * best_match_prev.pd_box[i_coord] + 
                        (1 - self.params['detection_smoothing_alpha']) * current_new_region.pd_box[i_coord]
                    )
    
    def _process_landmarks_and_gestures(self, regions, resized_frame):
        """Process landmarks and gestures exactly like your notebook"""
        processed_regions = []
        
        if not (regions and self.params['show_landmarks']):
            return processed_regions
        
        # Get compiled models
        compiled_model_landmark = self.model_manager.get_compiled_model('hand_landmarks')
        compiled_model_gesture = self.model_manager.get_compiled_model('gesture_embedder')
        compiled_model_classifier = self.model_manager.get_compiled_model('gesture_classifier')
        
        if not compiled_model_landmark:
            return processed_regions
        
        for region_idx, region_to_process in enumerate(regions):
            if not hasattr(region_to_process, 'rect_points'):
                continue
                
            try:
                # Process exactly like your notebook
                hand_crop_bgr = warp_rect_img(region_to_process.rect_points, resized_frame, 224, 224)
                hand_crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
                hand_input = np.expand_dims(hand_crop_rgb, axis=0).astype(np.float32) / 255.0
                
                lm_results = compiled_model_landmark([hand_input])
                
                if self.params['show_static_gestures']:
                    lm_postprocess_with_gesture_classification(
                        region_to_process, lm_results, 
                        self.params['previous_frame_processed_regions'],
                        compiled_model_gesture,
                        compiled_model_classifier,
                        alpha=self.params['smoothing_alpha'], 
                        iou_threshold=self.params['iou_match_threshold']
                    )
                    
                    self._apply_gesture_smoothing(region_to_process, region_idx)
                else:
                    lm_postprocess(region_to_process, lm_results, 
                                 self.params['previous_frame_processed_regions'],
                                 alpha=self.params['smoothing_alpha'], 
                                 iou_threshold=self.params['iou_match_threshold'])
                
                # Process finger detection exactly like your notebook
                # This call is now valid because of the import fix at the top of the file.
                process_finger_detection(region_to_process, self.params)
                
                processed_regions.append(region_to_process)
                
            except Exception as e:
                print(f"Error processing region {region_idx}: {e}")
        
        return processed_regions
    
    def _apply_gesture_smoothing(self, region, region_idx):
        """Apply gesture smoothing exactly like your notebook"""
        if not hasattr(region, 'gesture_name'):
            return
        
        region_id = f"region_{region_idx}"
        
        if region_id not in self.params['gesture_history']:
            self.params['gesture_history'][region_id] = []
        
        self.params['gesture_history'][region_id].append(region.gesture_name)
        if len(self.params['gesture_history'][region_id]) > self.params['gesture_smoothing_frames']:
            self.params['gesture_history'][region_id].pop(0)
        
        if len(self.params['gesture_history'][region_id]) >= 3:
            gesture_counts = {}
            for gest in self.params['gesture_history'][region_id]:
                gesture_counts[gest] = gesture_counts.get(gest, 0) + 1
            
            most_common_gesture = max(gesture_counts, key=lambda k: gesture_counts[k])
            if gesture_counts[most_common_gesture] >= len(self.params['gesture_history'][region_id]) // 2 + 1:
                region.gesture_name = most_common_gesture
    
    def _process_application_modes(self, region):
        """Delegate to application mode manager"""
        if self.app_mode_manager:
            self.app_mode_manager.process_application_modes(region)
    
    def _reset_hand_tracking(self, processed_regions):
        """Reset hand tracking exactly like your notebook"""
        any_gesture = any(hasattr(region, 'gesture_type') and region.gesture_type != "none" 
                          for region in processed_regions)
        
        if not any_gesture:
            self.params['last_pressed_hand'] = None
    
    def _render_results_complete(self, frame, processed_regions, frame_w, frame_h):
        """Render results with COMPLETE visual display exactly like your notebook"""
        if not processed_regions:
            return
        
        input_size = self.params['input_size']
        
        for region in processed_regions:
            if not hasattr(region, 'rect_points'):
                continue
                
            # Scale points to original frame
            scaled_points = []
            for ptx, pty in region.rect_points:
                scaled_ptx = int(ptx * frame_w / input_size)
                scaled_pty = int(pty * frame_h / input_size)
                scaled_points.append((scaled_ptx, scaled_pty))
            
            # Draw bounding rectangle
            points_array = np.array(scaled_points, np.int32)
            cv2.polylines(frame, [points_array], True, (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(frame, f"{region.pd_score:.2f}", 
                       (scaled_points[0][0], scaled_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Render landmarks with COMPLETE detail exactly like your notebook
            if self.params['show_landmarks'] and hasattr(region, 'landmarks'):
                original_rp_backup = region.rect_points
                region.rect_points = scaled_points
                lm_render(frame, region)
                region.rect_points = original_rp_backup
                
                # Display COMPLETE gesture info exactly like your notebook
                self._render_complete_gesture_info(frame, region, scaled_points)
    
    def _render_complete_gesture_info(self, frame, region, scaled_points):
        """Render COMPLETE gesture information exactly like your notebook"""
        if not (hasattr(region, 'gesture_type') and hasattr(region, 'hand_type')):
            return
        
        y_offset = 30
        
        # Show hand type exactly like notebook
        hand_text = f"{region.hand_type.upper()}"
        hand_color = (255, 0, 0) if region.hand_type == "left" else (0, 0, 255)
        cv2.putText(frame, hand_text, 
                   (scaled_points[0][0], scaled_points[0][1] + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        y_offset += 20
        
        # Show finger angles exactly like notebook
        if hasattr(region, 'index_angle') and hasattr(region, 'middle_angle'):
            angle_text = f"I:{region.index_angle:.0f}° M:{region.middle_angle:.0f}°"
            cv2.putText(frame, angle_text, 
                       (scaled_points[0][0], scaled_points[0][1] + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20
        
        # Show finger relationship exactly like notebook
        if hasattr(region, 'finger_angle_between'):
            relationship_text = ""
            relationship_color = (255, 255, 255)
            
            if hasattr(region, 'fingers_parallel') and region.fingers_parallel:
                relationship_text = f"PARALLEL ({region.finger_angle_between:.0f}°)"
                relationship_color = (0, 255, 255)  # Cyan for parallel
            elif hasattr(region, 'fingers_perpendicular') and region.fingers_perpendicular:
                relationship_text = f"PERPENDICULAR ({region.finger_angle_between:.0f}°)"
                relationship_color = (255, 0, 255)  # Magenta for perpendicular
            else:
                relationship_text = f"ANGLED ({region.finger_angle_between:.0f}°)"
                relationship_color = (128, 128, 128)  # Gray for other angles
            
            cv2.putText(frame, relationship_text, 
                       (scaled_points[0][0], scaled_points[0][1] + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, relationship_color, 2)
            y_offset += 20
        
        # Show finger states exactly like notebook
        if hasattr(region, 'middle_state'):
            state_text = f"I:{region.index_state} M:{region.middle_state}"
            cv2.putText(frame, state_text, 
                       (scaled_points[0][0], scaled_points[0][1] + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 20
        
        # Show gesture type exactly like notebook
        if region.gesture_type == "index_only":
            if hasattr(region, 'fingers_perpendicular') and region.fingers_perpendicular:
                gesture_text = "INDEX ONLY (⊥)"  # Perpendicular symbol
            else:
                gesture_text = "INDEX ONLY"
            gesture_color = (0, 255, 0)  # Green
        elif region.gesture_type == "index_middle_both":
            gesture_text = "INDEX + MIDDLE (∥)"  # Parallel symbol
            gesture_color = (0, 255, 255)  # Cyan
        else:
            gesture_text = "NO GESTURE"
            gesture_color = (128, 128, 128)  # Gray
        
        cv2.putText(frame, gesture_text, 
                   (scaled_points[0][0], scaled_points[0][1] + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        y_offset += 20
        
        # Show MediaPipe gesture exactly like notebook
        if hasattr(region, 'gesture_name'):
            mp_text = f"MP: {region.gesture_name}"
            if hasattr(region, 'gesture_confidence'):
                mp_text += f" ({region.gesture_confidence:.2f})"
            cv2.putText(frame, mp_text,
                       (scaled_points[0][0], scaled_points[0][1] + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)
    
    def _render_complete_status_info(self, frame, processed_regions, need_palm_detection):
        """Render ALL status information exactly like your notebook"""
        # Gesture mapping status exactly like notebook
        gesture_mapping_status = f"Gesture Mapping: {'ON' if self.params['gesture_mapping']['enable_gesture_mapping'] else 'OFF'}"
        gesture_mapping_color = (0, 255, 0) if self.params['gesture_mapping']['enable_gesture_mapping'] else (0, 0, 255)
        cv2.putText(frame, gesture_mapping_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_mapping_color, 2)
        
        # Application mode status exactly like notebook
        current_mode = self.app_modes['current_mode']
        if current_mode != 'disabled' and current_mode in self.app_modes:
            mode_text = f"App Mode: {self.app_modes[current_mode]['name']}"
            mode_color = (0, 255, 255)  # Cyan for active mode
            
            # Add browser sub-mode info exactly like notebook
            if current_mode == 'browser_mode':
                right_mode = self.app_modes['browser_mode']['right_hand_mode']
                mode_text += f" (Right: {right_mode.upper()})"
        else:
            mode_text = "App Mode: DISABLED"
            mode_color = (128, 128, 128)  # Gray for disabled
        
        cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Palm detection state exactly like notebook
        state_color_map = {
            'NO_HANDS': (255, 255, 0),          # Yellow
            'ONE_HAND_SEARCHING': (0, 255, 0),  # Green
            'ONE_HAND_STABLE': (0, 200, 0),     # Dark Green
            'TWO_HANDS': (255, 0, 255)          # Magenta
        }
        
        state_text = f"Palm State: {self.params['palm_detection_state']}"
        
        # Add countdown timer exactly like notebook
        if self.params['palm_detection_state'] == 'ONE_HAND_SEARCHING':
            remaining = self.params['grace_period_duration'] - (time.time() - self.params['grace_period_start'])
            if remaining > 0:
                state_text += f" ({remaining:.1f}s)"
        
        # Add hand count info exactly like notebook
        hand_count = len(processed_regions)
        state_text += f" | Hands: {hand_count}"
        
        cv2.putText(frame, state_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   state_color_map.get(self.params['palm_detection_state'], (255, 255, 255)), 2)
        
        # Detection mode info exactly like notebook
        detection_mode = "SMART" if not self.params['always_run_palm_detection'] else "ALWAYS"
        detection_active = "ACTIVE" if need_palm_detection else "IDLE"
        detection_info = f"Detection: {detection_mode} ({detection_active})"
        cv2.putText(frame, detection_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if need_palm_detection else (128, 128, 128), 2)
        
        # FPS info
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def get_status(self):
        """Get engine status"""
        return {
            'running': self.running,
            'paused': self.paused,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'models_loaded': self.model_manager.is_initialized(),
            'camera_active': self.cap is not None and self.cap.isOpened()
        }
    def switch_mode(self, mode_name: str):
        """Switch to a new application mode - delegate to manager"""
        if self.app_mode_manager:
            return self.app_mode_manager.switch_mode(mode_name)
        return False

# Global engine instance
complete_engine = CompleteGestureEngine()