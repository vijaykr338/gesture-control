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
from gesture_processor import process_finger_detection
from hand_landmark import *
from application_modes import ApplicationModeManager
import pyautogui
from game_controller import get_game_controller
import time


class CompleteGestureEngine:
    """Complete gesture detection engine with full visual rendering like your notebook"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.model_manager = model_manager
        
        
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
        
    def initialize(self, benchmark_mode: bool = False) -> bool:
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
        
        # Initialize models
        model_paths = {
            'palm_detection': 'mediapipeModels/hand_detector.xml',
            'hand_landmarks': 'mediapipeModels/hand_landmarks_detector.xml',
            'gesture_embedder': 'mediapipeModels/gesture_embedder.xml',
            'gesture_classifier': 'mediapipeModels/canned_gesture_classifier.xml'
        }
        
        if not self.model_manager.initialize_models(model_paths):
            print("❌ Model initialization failed!")
            return False
        
        # --- FIX: Skip camera initialization in benchmark mode ---
        if not benchmark_mode:
            # Initialize camera with better handling (only for normal mode)
            self.cap = None
            for camera_id in [0, 1, -1]:  # Try different camera indices
                try:
                    test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                    if test_cap.isOpened():
                        # Test if we can actually read frames
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera {camera_id} working!")
                            # Set optimal properties
                            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
        else:
            print("🔧 Benchmark mode: Skipping camera initialization")
            self.cap = None
        # --- END OF FIX ---
        
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

    def process_single_frame_benchmark(self, frame: np.ndarray):
        """
        Processes a single frame for benchmarking, returning the annotated frame and performance timings.
        This method does NOT use the camera and does NOT trigger application mode actions.
        """
        timings = {}
        overall_start_time = time.perf_counter()
        
        try:
            original_frame = frame.copy()
            frame_h, frame_w = original_frame.shape[:2]
            resized_frame_for_input = cv2.resize(frame, (self.params['input_size'], self.params['input_size']))

            # --- Palm Detection ---
            pd_start_time = time.perf_counter()
            
            # --- FIX: Use a more robust check for when to run palm detection ---
            # Palm detection is needed if the 'always run' flag is set, OR if tracking
            # is unstable (i.e., should_run_palm_detection returns True).
            always_run = self.params.get('always_run_palm_detection', False)
            tracking_requires_detection = should_run_palm_detection(
                self.params.get('previous_frame_processed_regions', []),
                self.params.get('landmark_score_for_palm_redetection_threshold', 0.7)
            )
            need_palm_detection = always_run or tracking_requires_detection
            # --- END OF FIX ---
            
            current_regions_for_processing = []
            if need_palm_detection:
                regions_nms = self._run_palm_detection(resized_frame_for_input)
                self._smooth_detection_boxes(regions_nms)
                current_regions_for_processing = regions_nms
            else:
                # Use the tracked regions from the previous frame
                current_regions_for_processing = self.params.get('previous_frame_processed_regions', [])

            timings['palm_detection_inference_ms'] = (time.perf_counter() - pd_start_time) * 1000

            if current_regions_for_processing:
                detections_to_rect(current_regions_for_processing)
                rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            
            # --- Landmark Processing ---
            lm_start_time = time.perf_counter()
            processed_regions = self._process_landmarks_and_gestures(current_regions_for_processing, resized_frame_for_input)
            timings['landmark_inference_ms'] = (time.perf_counter() - lm_start_time) * 1000
            
            # --- Rendering (for visual feedback) ---
            self._render_results_complete(original_frame, processed_regions, frame_w, frame_h)
            
            # Update previous regions for the next frame in the benchmark sequence
            self.params['previous_frame_processed_regions'] = list(processed_regions)
            
            timings['total_engine_time_ms'] = (time.perf_counter() - overall_start_time) * 1000
            
            return original_frame, timings

        except Exception as e:
            print(f"Error in benchmark frame processing: {e}")
            import traceback
            traceback.print_exc()
            return frame, {'error': str(e)}

    
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
            frame_h, frame_w = original_frame.shape[:2]
            resized_frame_for_input = cv2.resize(frame, (self.params['input_size'], self.params['input_size']))

            # Smart palm detection state machine
            current_hand_count = len(self.params['previous_frame_processed_regions'])

            # --- FORCE PALM DETECTION IN GAME MODE UNTIL BOTH HANDS FOUND ---
            in_game_mode = (self.app_mode_manager.app_modes.current_mode == 'game_mode')
            if in_game_mode and current_hand_count < 2:
                need_palm_detection = True
            else:
                need_palm_detection = self._smart_palm_detection_state_machine(current_hand_count)
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
            # NEW: Render game controller overlay if active
            self._render_game_controller_overlay(original_frame)
            
            # Display ALL status information exactly like notebook
            #self._render_complete_status_info(original_frame, processed_regions, need_palm_detection)
            
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
        """Render bounding boxes, landmarks, show gesture name, user-friendly gesture, and the actual action performed in the current mode as an overlay, plus FPS overlay."""
        if not processed_regions:
            return
        
        input_size = self.params['input_size']
        current_mode = None
        mode_action_map = None
        # Try to get current mode and gesture-action mapping
        if hasattr(self, 'app_modes') and hasattr(self.app_modes, 'current_mode'):
            current_mode = getattr(self.app_modes, 'current_mode', None)
            mode_config = getattr(self.app_modes, current_mode, None)
            if mode_config and hasattr(mode_config, 'gestures'):
                mode_action_map = mode_config.gestures
        
        for region in processed_regions:
            if not hasattr(region, 'rect_points'):
                continue
                
            # Scale points to original frame
            scaled_points = []
            for ptx, pty in region.rect_points:
                scaled_ptx = int(ptx * frame_w / input_size)
                scaled_pty = int(pty * frame_h / input_size)
                scaled_points.append((scaled_ptx, scaled_pty))
            
            # Draw bounding rectangle (keep)
            points_array = np.array(scaled_points, np.int32)
            cv2.polylines(frame, [points_array], True, (0, 255, 0), 2)
            
            # Draw landmarks (keep)
            if self.params['show_landmarks'] and hasattr(region, 'landmarks'):
                original_rp_backup = region.rect_points
                region.rect_points = scaled_points
                lm_render(frame, region)
                region.rect_points = original_rp_backup
            
            # Show gesture name and user-friendly gesture below the bounding box
            y_text = max(pt[1] for pt in scaled_points) + 25
            x_text = min(pt[0] for pt in scaled_points)
            gesture_name = getattr(region, 'gesture_name', None)
            gesture_type = getattr(region, 'gesture_type', None)
            # Map gesture_type to user-friendly action
            user_friendly_gesture = None
            if gesture_type == 'index_bent':
                user_friendly_gesture = 'Bend INDEX finger'
            elif gesture_type == 'index_middle_bent':
                user_friendly_gesture = 'Bend INDEX + MIDDLE fingers'
            elif gesture_type == 'fist':
                user_friendly_gesture = 'Make a FIST'
            elif gesture_type == 'open_palm':
                user_friendly_gesture = 'Show OPEN PALM'
            elif gesture_type == 'iloveyou':
                user_friendly_gesture = 'I LOVE YOU sign'
            # Add more mappings as needed
            else:
                user_friendly_gesture = gesture_type if gesture_type else ''
            gesture_text = f"Gesture: {gesture_name if gesture_name else 'No Gesture'}"
            cv2.putText(frame, gesture_text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if user_friendly_gesture:
                cv2.putText(frame, f"Detected: {user_friendly_gesture}", (x_text, y_text + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            # Show mode-specific action as overlay (actual action, not gesture name)
            mode_action = None
            if mode_action_map:
                # Try to find the action for this gesture in the current mode
                # Try both gesture_name and gesture_type as keys
                if gesture_name and gesture_name in mode_action_map:
                    mode_action = getattr(mode_action_map[gesture_name], 'action', None)
                elif gesture_type and gesture_type in mode_action_map:
                    mode_action = getattr(mode_action_map[gesture_type], 'action', None)
            if mode_action:
                cv2.putText(frame, f"Mode Action: {mode_action}", (x_text, y_text + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        # FPS overlay (top-left corner)
        fps_val = self.fps if hasattr(self, 'fps') else 0
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def _render_game_controller_overlay(self, frame):
        """Renders the steering box and other game control visuals."""
        game_controller = get_game_controller()
        if not game_controller or not game_controller.active:
            return

        overlay_data = game_controller.get_steering_box_data()
        if not overlay_data:
            return

        box = overlay_data['box']
        
        # Draw main steering box
        cv2.rectangle(frame, 
                      (int(box['left']), int(box['top'])), 
                      (int(box['right']), int(box['bottom'])), 
                      (255, 255, 0), 2) # Cyan box

        # Draw deadzone
        cv2.line(frame, 
                 (int(overlay_data['deadzone_left']), int(box['top'])), 
                 (int(overlay_data['deadzone_left']), int(box['bottom'])), 
                 (0, 0, 255), 1) # Red line
        cv2.line(frame, 
                 (int(overlay_data['deadzone_right']), int(box['top'])), 
                 (int(overlay_data['deadzone_right']), int(box['bottom'])), 
                 (0, 0, 255), 1) # Red line

        palm_center = overlay_data.get('palm_center')
        if palm_center:
            cv2.circle(frame, (int(palm_center[0]), int(palm_center[1])), 15, (0, 255, 255), -1) # Yellow filled circle
            cv2.putText(frame, "Center", 
                        (int(palm_center[0]) + 20, int(palm_center[1]) + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw steering indicator
        steering_angle = overlay_data['steering_angle']
        indicator_x = int(box['center_x'] + (steering_angle * (box['width'] / 2)))
        cv2.circle(frame, (indicator_x, int(box['bottom']) - 20), 15, (0, 255, 0), -1) # Green circle
        
        # Draw status text
        status_text = f"Steering: {steering_angle:.2f}"
        if overlay_data['is_accelerating']:
            status_text += " | ACCELERATING"
        
        cv2.putText(frame, status_text, 
                    (int(box['left']), int(box['top']) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Remove or comment out _render_complete_gesture_info and _render_complete_status_info calls in the main pipeline.
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
        print(f"🔄 Engine switching to mode: {mode_name}")
        if self.app_mode_manager:
            result = self.app_mode_manager.switch_mode(mode_name)
            print(f"   Mode switch result: {result}")
            
            # DEBUG: Check game controller state after switch
            if mode_name == 'game_mode':
                game_controller = get_game_controller()
                if game_controller:
                    print(f"   Game controller active after switch: {game_controller.active}")
                else:
                    print("   ❌ No game controller found after mode switch")
            
            return result
        return False

# Global engine instance
complete_engine = CompleteGestureEngine()