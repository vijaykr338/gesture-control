"""
Independent game controller module for racing games.
Only activated when Game Mode is selected - no conflicts with existing system.
"""

import time
import math
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import pydirectinput
    DIRECTINPUT_AVAILABLE = True
except ImportError:
    DIRECTINPUT_AVAILABLE = False
    print("⚠️ pydirectinput not available. Using keyboard fallback.")

class GameController:
    """Game control system with keyboard fallback (no external dependencies required)"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.active = False
        self.last_palm_center = None # Add this line
        
        # Game state
        self.steering_angle = 0.0
        self.is_accelerating = False
        self.last_steering_update = 0.0
        
        # Steering box boundaries
        self.steering_box = {}
        self._update_steering_box()
        
        # Key state tracking
        self.pressed_keys = set()
        self.active_action_keys = set() # Tracks held action keys (brake, nitrous)
        
        # State machine for steering
        self.steering_state = 'NEUTRAL'  # NEUTRAL, TURN_INITIATE, LIGHT_TURN, HEAVY_TURN, DIRECTION_CHANGE, TURN_RELEASE
        self.state_start_time = 0.0
        self.turn_direction = None  # 'LEFT', 'RIGHT', or None
        self.turn_intensity = 0.0
        self.last_hand_position = None
        self.steering_history = []  # For tracking hand movement patterns
    
    def activate(self):
        """Activate game controller"""
        self.active = True
        self._update_steering_box()
        # print("🎮 Game Controller ACTIVATED")

    def deactivate(self):
        """Deactivate and cleanup"""
        self.active = False
        self._release_all_keys()
        self.active_action_keys.clear()
        # Reset state machine
        self.steering_state = 'NEUTRAL'
        self.turn_direction = None
        self.turn_intensity = 0.0
        self.last_hand_position = None
        # print("🎮 Game Controller DEACTIVATED")

    def _update_steering_box(self):
        """Update steering box boundaries from params"""
        screen_w = self.params.get('screen_width', 1920)
        screen_h = self.params.get('screen_height', 1080)
        box_w = self.params.get('steering_box_width', 0.4) * screen_w
        box_h = self.params.get('steering_box_height', 0.3) * screen_h
        box_x = self.params.get('steering_box_x', 0.3) * screen_w
        box_y = self.params.get('steering_box_y', 0.35) * screen_h
        self.steering_box = {
            'left': box_x, 'right': box_x + box_w, 'top': box_y, 'bottom': box_y + box_h,
            'center_x': box_x + box_w / 2, 'width': box_w, 'height': box_h
        }
        # Commented out debug print
        # print(f"🎮 Steering box updated:")
        # print(f"   Screen: {screen_w}x{screen_h}")
        # print(f"   Box: {box_x:.1f},{box_y:.1f} to {box_x + box_w:.1f},{box_y + box_h:.1f}")
        # print(f"   Size: {box_w:.1f}x{box_h:.1f}")
        # print(f"   Params: width={self.params.get('steering_box_width', 'missing')}, height={self.params.get('steering_box_height', 'missing')}")
    
    def _get_transformed_landmark(self, region, landmark_index: int) -> Optional[Tuple[float, float]]:
        """Transforms a single landmark from its local crop-space to the full screen-space."""
        if not hasattr(region, 'rect_points') or not hasattr(region, 'landmarks') or len(region.landmarks) <= landmark_index:
            return None

        rect_points = region.rect_points
        landmark_crop_norm = region.landmarks[landmark_index]
        lm_xy_crop_pixels = np.array([(landmark_crop_norm[0] * 224.0, landmark_crop_norm[1] * 224.0)], dtype=np.float32)
        src_crop_coords = np.array([(0, 0), (224, 0), (224, 224)], dtype=np.float32)
        dst_rect_coords = np.array([rect_points[1], rect_points[2], rect_points[3]], dtype=np.float32)
        mat = cv2.getAffineTransform(src_crop_coords, dst_rect_coords)
        lm_xy_transformed = cv2.transform(np.expand_dims(lm_xy_crop_pixels, axis=0), mat)
        return tuple(np.squeeze(lm_xy_transformed).astype(np.int32))

    def _get_palm_center_screen_coords(self, region) -> Optional[Tuple[float, float]]:
        """
        Returns the screen coordinates for the middle finger base landmark (point 9).
        Amplifies displacement from the center of the steering box, then applies a 20% right and 40% down offset.
        """
        if not hasattr(region, 'landmarks') or not hasattr(region, 'rect_points') or len(region.landmarks) <= 9:
            return None
            
        rect_points = region.rect_points
        lm_xy_crop_pixels = np.array([(l[0] * 224.0, l[1] * 224.0) for l in region.landmarks], dtype=np.float32)
        src_crop_coords = np.array([(0, 0), (224, 0), (224, 224)], dtype=np.float32)
        dst_rect_coords = np.array([rect_points[1], rect_points[2], rect_points[3]], dtype=np.float32)
        mat = cv2.getAffineTransform(src_crop_coords, dst_rect_coords)
        lm_xy_transformed = cv2.transform(np.expand_dims(lm_xy_crop_pixels, axis=0), mat)
        lm_xy_final = np.squeeze(lm_xy_transformed).astype(np.int32)
        
        # Get landmark #9 (middle finger base) coordinates
        raw_screen_x, raw_screen_y = lm_xy_final[8].astype(float)

        # Amplify the displacement from the center of the steering box
        amplification = self.params.get('steering_displacement_amplification', 1.0)
        if amplification != 1.0 and 'center_x' in self.steering_box:
            box_center_x = self.steering_box['center_x']
            delta_x = raw_screen_x - box_center_x
            amplified_delta_x = delta_x * amplification
            screen_x = box_center_x + amplified_delta_x
        else:
            screen_x = raw_screen_x

        # Apply 20% right and 40% down offset
        screen_width = self.params.get('screen_width', 1920)
        screen_height = self.params.get('screen_height', 1080)
        screen_x = screen_x + (screen_width - screen_x) * 0.18
        screen_y = raw_screen_y + (screen_height - raw_screen_y) * 0.28

        return (screen_x, screen_y)

    def handle_right_hand_steering(self, region, is_open_palm: bool):
        """Handle right hand steering for any detected hand (no gesture restriction)."""
        if not self.active:
            self.last_palm_center = None # Reset on inactive
            return

        palm_center_px = self._get_palm_center_screen_coords(region)
        self.last_palm_center = palm_center_px # Store the calculated center
        if palm_center_px is None:
            self._release_key('left')
            self._release_key('right')
            return

        screen_x, screen_y = palm_center_px
        box = self.steering_box
        in_box = self._is_in_steering_box(screen_x, screen_y)
        if not in_box:
            self._release_key('left')
            self._release_key('right')
            return

        old_angle = self.steering_angle
        self._calculate_steering_angle(screen_x)
        self._apply_controls()

    def update_left_hand_actions(self, detected_gestures_this_frame: list):
        """Updates the state of held keys based on the gestures detected in the current frame."""
        if not self.active: return

        action_map = {
            "left_index_middle_bent": "up",   # Acceleration
            "fist_gesture": "shift",          # Nitrous
            # Add other mappings as needed
        }

        required_keys_now = set()
        # If nitrous is active, also require acceleration
        if "fist_gesture" in detected_gestures_this_frame:
            required_keys_now.add("shift")
            required_keys_now.add("up")  # Hold acceleration with nitrous
        if "left_index_middle_bent" in detected_gestures_this_frame:
            required_keys_now.add("up")

        # Release keys that are no longer needed
        keys_to_release = self.active_action_keys - required_keys_now
        for key in keys_to_release:
            self._release_key(key)

        # Press keys that are now required
        keys_to_press = required_keys_now - self.active_action_keys
        for key in keys_to_press:
            self._press_key(key)

        self.active_action_keys = required_keys_now
    
    def _is_in_steering_box(self, x: float, y: float) -> bool:
        """Check if position is within steering box"""
        return (self.steering_box['left'] <= x <= self.steering_box['right'] and
                self.steering_box['top'] <= y <= self.steering_box['bottom'])
    
    def _calculate_steering_angle(self, screen_x: float):
        """Calculate steering angle and update state machine"""
        current_time = time.time()
        
        # Calculate raw steering position
        relative_x = (screen_x - self.steering_box['left']) / self.steering_box['width']
        deadzone = self.params.get('steering_deadzone', 0.1)
        
        target_angle = 0.0
        if relative_x < 0.5 - deadzone:  # Left side
            target_angle = (relative_x - (0.5 - deadzone)) / (0.5 - deadzone)
        elif relative_x > 0.5 + deadzone:  # Right side
            target_angle = (relative_x - (0.5 + deadzone)) / (0.5 - deadzone)
        
        # Apply sensitivity and limits
        sensitivity = self.params.get('steering_sensitivity', 1.0)
        target_angle *= sensitivity
        target_angle = max(-1.0, min(1.0, target_angle))
        
        # Store for state machine
        self.steering_angle = target_angle
        self.last_hand_position = screen_x
        
        # Add to history for movement tracking
        self.steering_history.append((current_time, target_angle))
        if len(self.steering_history) > 5:  # Keep last 5 samples
            self.steering_history.pop(0)
        
        # Update state machine
        self._update_steering_state_machine(current_time)
    
    def _start_acceleration(self):
        """Start acceleration (no longer called from steering)"""
        if not self.is_accelerating:
            self.is_accelerating = True
            self._press_key('up')

    def _stop_acceleration(self):
        """Stop acceleration and steering"""
        if self.is_accelerating:
            self.is_accelerating = False
            self.steering_angle = 0.0
            self._release_key('up')
            self._release_key('left')
            self._release_key('right')
    
    def _apply_controls(self):
        """Apply steering controls using state machine - called after _calculate_steering_angle"""
        # State machine handles all steering logic now
        # No direct key control here - everything goes through state machine
        pass
    
    def _update_steering_state_machine(self, current_time: float):
        """State machine for consistent steering behavior"""
        abs_angle = abs(self.steering_angle)
        direction = 'LEFT' if self.steering_angle > 0 else 'RIGHT'
        state_duration = current_time - self.state_start_time
        
        # Configuration
        deadzone_threshold = 0.15
        light_turn_threshold = 0.4
        heavy_turn_threshold = 0.7
        
        if self.steering_state == 'NEUTRAL':
            if abs_angle > deadzone_threshold:
                self._transition_to_state('TURN_INITIATE', current_time)
                self.turn_direction = direction
                self.turn_intensity = abs_angle
                
        elif self.steering_state == 'TURN_INITIATE':
            if state_duration > 0.05:  # 50ms confirmation period
                if abs_angle > deadzone_threshold and direction == self.turn_direction:
                    # Confirmed turn - decide intensity
                    if abs_angle < light_turn_threshold:
                        self._transition_to_state('LIGHT_TURN', current_time)
                    else:
                        self._transition_to_state('HEAVY_TURN', current_time)
                    if self.turn_direction:  # Safety check
                        self._start_turn(self.turn_direction)
                else:
                    # False alarm - back to neutral
                    self._transition_to_state('NEUTRAL', current_time)
                    
        elif self.steering_state in ['LIGHT_TURN', 'HEAVY_TURN']:
            current_direction = 'LEFT' if self.steering_angle > 0 else 'RIGHT'
            
            # Check for direction change
            if current_direction != self.turn_direction and abs_angle > 0.2:
                self._transition_to_state('DIRECTION_CHANGE', current_time)
                self._stop_turn()
                return
                
            # Check for return to neutral
            if abs_angle < 0.1:
                self._transition_to_state('TURN_RELEASE', current_time)
                self._stop_turn()
                return
                
            # Update turn intensity if needed
            if abs_angle > heavy_turn_threshold and self.steering_state == 'LIGHT_TURN':
                self.steering_state = 'HEAVY_TURN'
                self.turn_intensity = abs_angle
            elif abs_angle < light_turn_threshold and self.steering_state == 'HEAVY_TURN':
                self.steering_state = 'LIGHT_TURN' 
                self.turn_intensity = abs_angle
                
            # Apply pulsed steering based on intensity
            self._apply_pulsed_steering(current_time)
            
        elif self.steering_state == 'DIRECTION_CHANGE':
            if state_duration > 0.1:  # 100ms pause for direction change
                self._transition_to_state('NEUTRAL', current_time)
                
        elif self.steering_state == 'TURN_RELEASE':
            if state_duration > 0.05:  # 50ms release period
                self._transition_to_state('NEUTRAL', current_time)
        
        # Emergency timeout - return to neutral if stuck in any state too long
        if state_duration > 2.0:
            self._transition_to_state('NEUTRAL', current_time)
            self._stop_turn()
    
    def _transition_to_state(self, new_state: str, current_time: float):
        """Transition to a new steering state"""
        # print(f"🎮 Steering: {self.steering_state} → {new_state}")
        self.steering_state = new_state
        self.state_start_time = current_time
    
    def _start_turn(self, direction: str):
        """Start turning in specified direction"""
        key = 'left' if direction == 'LEFT' else 'right'
        opposite = 'right' if direction == 'LEFT' else 'left'
        
        # Release opposite direction immediately
        if opposite in self.pressed_keys:
            self._release_key(opposite)
        
        # Start turning
        if key not in self.pressed_keys:
            self._press_key(key)
    
    def _stop_turn(self):
        """Stop all turning"""
        self._release_key('left')
        self._release_key('right')
    
    def _apply_pulsed_steering(self, current_time: float):
        """Apply pulsed steering based on current intensity"""
        if not hasattr(self, 'last_pulse_time'):
            self.last_pulse_time = 0
            self.in_pulse = False
        
        # Determine pulse timing based on intensity
        if self.steering_state == 'LIGHT_TURN':
            pulse_duration = 0.08  # Short pulses for light turns
            pause_duration = 0.12
        else:  # HEAVY_TURN
            pulse_duration = 0.15  # Longer pulses for heavy turns
            pause_duration = 0.05
        
        time_since_pulse = current_time - self.last_pulse_time
        
        if not self.in_pulse and time_since_pulse >= pause_duration:
            # Start new pulse
            if self.turn_direction:  # Safety check
                self._start_turn(self.turn_direction)
            self.in_pulse = True
            self.last_pulse_time = current_time
            
        elif self.in_pulse and time_since_pulse >= pulse_duration:
            # End current pulse
            self._stop_turn()
            self.in_pulse = False
            self.last_pulse_time = current_time
    
    def _press_key(self, key: str):
        """Press and hold a key"""
        if key not in self.pressed_keys:
            try:
                if DIRECTINPUT_AVAILABLE:
                    pydirectinput.keyDown(key)
                else:
                    import pyautogui
                    # print(f"⚠️ Using pyautogui for keyDown: {key}")
                    pyautogui.keyDown(key)
                self.pressed_keys.add(key)
                # print(f"✅ Key DOWN: {key}")
            except Exception as e:
                print(f"❌ Error pressing key {key}: {e}")

    def _release_key(self, key: str):
        """Release a held key"""
        if key in self.pressed_keys:
            try:
                if DIRECTINPUT_AVAILABLE:
                    pydirectinput.keyUp(key)
                else:
                    import pyautogui
                    # print(f"⚠️ Using pyautogui for keyUp: {key}")
                    pyautogui.keyUp(key)
                self.pressed_keys.discard(key)
                # print(f"⬆️ Key UP: {key}")
            except Exception as e:
                print(f"❌ Error releasing key {key}: {e}")
    

    
    def _release_all_keys(self):
        """Release all held keys"""
        keys_to_release = list(self.pressed_keys)
        for key in keys_to_release: self._release_key(key)
        self.pressed_keys.clear()
    
    def get_steering_box_data(self) -> Dict[str, Any]:
        """Get data for rendering steering box overlay"""
        if not self.active: return {}
        box = self.steering_box
        deadzone_width = box['width'] * self.params.get('steering_deadzone', 0.1)
        return {
            'box': box, 'steering_angle': self.steering_angle, 'is_accelerating': self.is_accelerating,
            'deadzone_left': box['center_x'] - deadzone_width / 2,
            'deadzone_right': box['center_x'] + deadzone_width / 2,
            'palm_center': self.last_palm_center # Add this line
        }

_game_controller = None
def get_game_controller(params: Optional[Dict[str, Any]] = None) -> Optional[GameController]:
    """Get or create game controller instance"""
    global _game_controller
    if _game_controller is None and params:
        _game_controller = GameController(params)
    return _game_controller