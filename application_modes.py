import time
import pyautogui
import numpy as np
import math
from typing import Dict, Any
from config_manager import ApplicationModeGesture, config_manager

class ApplicationModeManager:
    """Manages application-specific gesture modes and actions using dataclass objects."""
    
    def __init__(self, app_modes_config: Any):
        # Defensively ensure self.app_modes is the dataclass object, not a raw dict.
        if isinstance(app_modes_config, dict):
            self.app_modes = config_manager.app_modes
        else:
            self.app_modes = app_modes_config
        
        self.params = {}  # Will be set by engine
        
    def set_engine_params(self, params: Dict[str, Any]):
        """Set reference to engine parameters"""
        self.params = params

    def process_application_modes(self, region):
        """Main processor for application modes, using dataclass attribute access."""
        if not hasattr(region, 'landmarks'):
            return
        
        self._handle_browser_mode_iloveyou(region)
        
        current_mode_key = self.app_modes.current_mode
        if current_mode_key == 'disabled':
            return
            
        mode_config = getattr(self.app_modes, current_mode_key, None)
        if not mode_config or not mode_config.enabled:
            return

        hand_type = "right" if region.handedness > 0.5 else "left"
        
        detected_gestures = []
        
        # FIXED: Properly map gesture types to gesture IDs
        if hasattr(region, 'gesture_type'):
            if region.gesture_type == "index_only":
                detected_gestures.append(f"{hand_type}_index_bent")
            elif region.gesture_type == "index_middle_both":
                detected_gestures.append(f"{hand_type}_index_middle_bent")
        
        # Check for MediaPipe static gestures
    #ADD STATIC GESTURE MAPPINGS
        if hasattr(region, 'gesture_name'):
            if region.gesture_name == "Closed_Fist":
                detected_gestures.append('fist_gesture')
            elif region.gesture_name == "Open_Palm":
                detected_gestures.append('open_palm_gesture')
            elif region.gesture_name == "ILoveYou":
                detected_gestures.append('iloveyou_gesture')
            # Add more MediaPipe gesture mappings as needed
            
        # Execute gestures that exist in the current mode
        for gesture_id in set(detected_gestures):
            if gesture_id in mode_config.gestures:
                gesture_data = mode_config.gestures[gesture_id]
                success = self._execute_application_gesture(gesture_id, gesture_data)
                if success:
                    print(f"🎯 Executed {gesture_id} in mode {current_mode_key}")
        
        # Browser mode special handling
        if current_mode_key == 'browser_mode' and hand_type == "right":
            right_mode = self.app_modes.browser_right_hand_mode
            if right_mode == 'cursor':
                self.handle_cursor_control(region, force_enable=True)
            elif right_mode == 'scroll':
                self.handle_scroll_control(region, force_enable=True)

    def switch_mode(self, mode_name: str) -> bool:
        """Switch to a new application mode using dataclass attribute access."""
        current_time = time.time()
        if current_time - self.app_modes.last_mode_switch < self.app_modes.mode_switch_cooldown:
            return False
        
        current_mode_key = self.app_modes.current_mode
        if current_mode_key != 'disabled':
            current_mode_obj = getattr(self.app_modes, current_mode_key, None)
            if current_mode_obj:
                current_mode_obj.enabled = False
        
        self.app_modes.current_mode = mode_name
        new_mode_obj = None
        if mode_name != 'disabled':
            new_mode_obj = getattr(self.app_modes, mode_name, None)
            if new_mode_obj:
                new_mode_obj.enabled = True
        
        self.app_modes.last_mode_switch = current_time
        
        mode_display = new_mode_obj.name if new_mode_obj else 'Disabled'
        print(f"🔄 MODE SWITCH: {mode_display}")
        return True

    def _handle_browser_mode_iloveyou(self, region):
        """ILoveYou gesture logic updated for dataclass attribute access."""
        if not self.app_modes.browser_mode.enabled:
            return
        if not hasattr(region, 'gesture_name') or region.gesture_name != "ILoveYou":
            return
        
        hand_type = "right" if region.handedness > 0.5 else "left"
        if hand_type != "right":
            return
        
        current_time = time.time()
        
        if current_time - self.app_modes.browser_last_iloveyou_switch < self.app_modes.browser_iloveyou_switch_cooldown:
            return
        
        current_rh_mode = self.app_modes.browser_right_hand_mode
        new_rh_mode = 'scroll' if current_rh_mode == 'cursor' else 'cursor'
        self.app_modes.browser_right_hand_mode = new_rh_mode
        self.app_modes.browser_last_iloveyou_switch = current_time
        print(f"🤟 BROWSER: Right hand mode → {new_rh_mode.upper()}")

    def _execute_application_gesture(self, gesture_id: str, gesture_config: ApplicationModeGesture):
        """Gesture execution logic with proper validation."""
        current_time = time.time()
        if current_time - self.app_modes.gesture_timings.get(gesture_id, 0) < gesture_config.cooldown:
            return False
        
        try:
            action = gesture_config.action
            if action == 'key_press' and gesture_config.key:
                # NEW: Handle key combinations
                if '+' in gesture_config.key:
                    keys_to_press = [k.strip() for k in gesture_config.key.split('+')]
                    pyautogui.hotkey(*keys_to_press)
                    print(f"🎯 {gesture_config.description}: Hotkey {gesture_config.key.upper()}")
                else:
                    pyautogui.press(gesture_config.key)
                    print(f"🎯 {gesture_config.description}: Key Press {gesture_config.key.upper()}")
            elif action == 'mouse_click' and gesture_config.button:
                # Validate mouse button before clicking
                valid_buttons = ('left', 'middle', 'right', 'primary', 'secondary')
                if gesture_config.button not in valid_buttons:
                    print(f"❌ Invalid mouse button '{gesture_config.button}' for gesture '{gesture_id}'. Skipping.")
                    return False
                pyautogui.click(button=gesture_config.button)
                print(f"🖱️ {gesture_config.description}: {gesture_config.button.upper()} CLICK")
            else:
                print(f"❌ Invalid gesture configuration for {gesture_id}: action='{action}', key='{gesture_config.key}', button='{gesture_config.button}'")
                return False
            
            self.app_modes.gesture_timings[gesture_id] = current_time
            return True
        except Exception as e:
            print(f"❌ Error executing gesture {gesture_id}: {e}")
            return False

    def handle_cursor_control(self, region, force_enable=False):
        """OPTIMIZED: Your cursor control logic, now non-blocking and efficient."""
        if not force_enable and not self.params.get('enable_cursor_control', False):
            return
        if not hasattr(region, 'landmarks') or len(region.landmarks) < 9:
            return
        
        try:
            index_tip = region.landmarks[8]
            
            target_x = self.params['screen_width'] * (1 - index_tip[0])
            target_y = self.params['screen_height'] * index_tip[1]
            
            if self.params.get('previous_cursor_pos') is not None:
                prev_x, prev_y = self.params['previous_cursor_pos']
                smooth_x = self.params['cursor_smoothing'] * prev_x + (1 - self.params['cursor_smoothing']) * target_x
                smooth_y = self.params['cursor_smoothing'] * prev_y + (1 - self.params['cursor_smoothing']) * target_y
            else:
                smooth_x, smooth_y = target_x, target_y
            
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            self.params['previous_cursor_pos'] = (smooth_x, smooth_y)
            
        except Exception as e:
            print(f"Error in cursor control: {e}")

    def handle_scroll_control(self, region, force_enable=False):
        """FIXED: Scroll control logic with proper gesture detection."""
        if not force_enable and not self.params.get('enable_scroll_control', False):
            return
        if not hasattr(region, 'landmarks') or len(region.landmarks) < 13:
            return
        
        try:
            index_tip = region.landmarks[8]
            middle_tip = region.landmarks[12]
            
            # FIXED: Calculate distance properly for scroll gesture detection
            finger_distance = math.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
            current_hand = "right" if region.handedness > 0.5 else "left"
            
            # Check hand preference
            scroll_hand_pref = self.params.get('scroll_hand_preference', 'any')
            if scroll_hand_pref != 'any' and current_hand != scroll_hand_pref:
                return
            
            # FIXED: Initialize scroll state properly
            if 'scroll_state' not in self.params:
                self.params['scroll_state'] = {
                    'is_scrolling': False, 
                    'start_pos': None, 
                    'locked_direction': None, 
                    'active_hand': None,
                    'last_scroll_time': 0
                }
            
            scroll_state = self.params['scroll_state']
            center_y = (index_tip[1] + middle_tip[1]) / 2
            current_time = time.time()
            
            # FIXED: Use proper scroll threshold (default to 0.02 if not set)
            scroll_threshold = self.params.get('scroll_threshold', 0.04)
            
            print(f"[DEBUG] Hand: {current_hand}, Distance: {finger_distance:.4f}, Threshold: {scroll_threshold:.4f}")
            
            # Check if scroll gesture is active (fingers close together)
            if finger_distance < scroll_threshold:
                if not scroll_state['is_scrolling']:
                    # Start scrolling
                    scroll_state.update({
                        'is_scrolling': True, 
                        'start_pos': center_y, 
                        'active_hand': current_hand,
                        'locked_direction': None
                    })
                    print(f"🔒 {current_hand.upper()} SCROLL START (threshold: {scroll_threshold:.3f})")
                    return
                
                # Continue scrolling only with the same hand that started
                if scroll_state['active_hand'] != current_hand:
                    return
                
                # Calculate movement from start position
                delta_from_start = center_y - scroll_state['start_pos']
                
                # Lock direction after sufficient movement
                if scroll_state['locked_direction'] is None and abs(delta_from_start) > 0.03:
                    scroll_state['locked_direction'] = "down" if delta_from_start > 0 else "up"
                    print(f"🎯 {current_hand.upper()} DIRECTION LOCKED: {scroll_state['locked_direction'].upper()}")
                
                # Perform scrolling if direction is locked
                if scroll_state['locked_direction'] is not None:
                    current_direction = "down" if delta_from_start > 0 else "up"
                    
                    # Only scroll in the locked direction
                    if current_direction == scroll_state['locked_direction']:
                        # FIXED: Add scroll throttling to prevent excessive scrolling
                        if current_time - scroll_state['last_scroll_time'] > 0.1:  # Limit to 10 scrolls per second
                            scroll_sensitivity = self.params.get('scroll_sensitivity', 6)
                            scroll_amount = int(-delta_from_start * scroll_sensitivity * 80)
                            
                            if abs(scroll_amount) > 2:  # Minimum scroll threshold
                                pyautogui.scroll(scroll_amount)
                                scroll_state['last_scroll_time'] = current_time
                                direction_arrow = "↑" if scroll_amount > 0 else "↓"
                                print(f"📜 {current_hand.upper()} SCROLL {direction_arrow} (amount: {scroll_amount})")
            else:
                # End scrolling when fingers move apart
                if scroll_state['is_scrolling']:
                    print(f"🔓 {scroll_state['active_hand'].upper()} SCROLL END")
                    scroll_state.update({
                        'is_scrolling': False, 
                        'start_pos': None, 
                        'locked_direction': None, 
                        'active_hand': None
                    })
                    
        except Exception as e:
            print(f"❌ Error in scroll control: {e}")
            import traceback
            traceback.print_exc()