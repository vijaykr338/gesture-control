import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStatusBar, 
                            QGroupBox, QComboBox, QSlider, QCheckBox, QSpinBox,
                            QScrollArea, QGridLayout, QMessageBox, QSplitter,
                            QTabWidget, QDoubleSpinBox, QListWidget, QStackedWidget,
                            QListWidgetItem, QFrame, QLineEdit, QStyle, QTextEdit, 
                            QDialog, QInputDialog, QButtonGroup, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt6.QtCore import Qt, QSize, QThread, QPoint, QRect, QMargins

from gui_worker import GestureEngineWorker
from config_manager import config_manager, ApplicationModeConfig, ApplicationModeGesture

class FlowLayout(QGridLayout):
    """A layout that arranges widgets in a flowing manner."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._item_list = []

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            wid = item.widget()
            space_x = spacing + wid.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Horizontal
            )
            space_y = spacing + wid.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Vertical
            )
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

class CustomModeDialog(QDialog):
    """Modal dialog for creating/editing custom gesture modes."""
    
    def __init__(self, parent=None, edit_mode=None, mode_data=None):
        super().__init__(parent)
        self.edit_mode = edit_mode
        self.mode_data = mode_data
        
        # Define valid keys and buttons separately
        self.valid_mouse_buttons = ["left", "right", "middle"]
        self.valid_keyboard_keys = config_manager.gesture_mapping.available_keys
        
        self.setWindowTitle("Custom Mode Builder" if not edit_mode else f"Edit {edit_mode.replace('_', ' ').title()}")
        self.setFixedSize(800, 600)
        self.setModal(True)
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
        
        self.setup_ui()
        if edit_mode and mode_data:
            self.load_existing_mode()

    def setup_ui(self):
        """Setup the custom mode dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        if self.edit_mode:
            header = QLabel(f"✏️ Editing {self.edit_mode.replace('_', ' ').title()}")
        else:
            header = QLabel("🛠️ Create Your Custom Gesture Mode")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ffff; margin-bottom: 10px;")
        layout.addWidget(header)

        # Mode Name (only for new modes)
        if not self.edit_mode:
            name_frame = QFrame()
            name_layout = QHBoxLayout(name_frame)
            name_layout.addWidget(QLabel("Mode Name:"))
            self.mode_name_edit = QLineEdit("My Custom Mode")
            name_layout.addWidget(self.mode_name_edit)
            layout.addWidget(name_frame)

        # Gestures Section
        gestures_group = QGroupBox("Configure Gestures")
        gestures_layout = QVBoxLayout(gestures_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        available_gestures = [
            'right_index_bent', 'left_index_bent', 
            'right_index_middle_bent', 'left_index_middle_bent', 
            'fist_gesture'
        ]
        
        self.gesture_widgets = {}
        for gesture_key in available_gestures:
            gesture_card = self.create_gesture_card(gesture_key)
            scroll_layout.addWidget(gesture_card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        gestures_layout.addWidget(scroll)
        layout.addWidget(gestures_group)

        # Action Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("❌ Cancel")
        save_btn = QPushButton("💾 Save Mode" if not self.edit_mode else "💾 Update Mode")
        
        cancel_btn.clicked.connect(self.reject)
        save_btn.clicked.connect(self.save_mode)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

    def _on_action_type_changed(self, gesture_key, is_key_press):
        """Called when action type changes - updates the key/button to a valid default."""
        widgets = self.gesture_widgets[gesture_key]
        
        if is_key_press:
            # Switched to Key Press - default to 'space'
            widgets['current_key'] = 'space'
            self.update_key_display(gesture_key, 'space')
        else:
            # Switched to Mouse Click - default to 'left'
            widgets['current_key'] = 'left'
            self.update_key_display(gesture_key, 'left')

    def create_gesture_card(self, gesture_key):
        """Create a card for configuring a single gesture."""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 1px solid #222222;
                border-radius: 8px;
                margin: 4px;
                padding: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)    

        gesture_name = self.get_gesture_display_name(gesture_key)
        name_label = QLabel(gesture_name)
        name_label.setStyleSheet("""
            font-weight: 700;
            font-size: 14px;
            color: #00ffff;
            padding: 6px;
            background-color: #000000;
            border-radius: 4px;
            margin-bottom: 8px;
        """)
        layout.addWidget(name_label)
        
        controls_layout = QGridLayout()
        
        enable_cb = QCheckBox("Enable this gesture")
        enable_cb.toggled.connect(lambda checked, key=gesture_key: self.toggle_gesture_controls(key, checked))
        controls_layout.addWidget(enable_cb, 0, 0, 1, 2)
        
        controls_layout.addWidget(QLabel("Description:"), 1, 0)
        desc_edit = QLineEdit()
        desc_edit.setEnabled(False)
        controls_layout.addWidget(desc_edit, 1, 1)
        
        controls_layout.addWidget(QLabel("Action Type:"), 2, 0)
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        
        key_press_btn = QPushButton("⌨️ Key Press")
        mouse_click_btn = QPushButton("🖱️ Mouse Click")
        
        key_press_btn.setCheckable(True)
        mouse_click_btn.setCheckable(True)
        key_press_btn.setChecked(True)
        
        # Connect action type change handlers
        key_press_btn.toggled.connect(lambda checked, key=gesture_key: self._on_action_type_changed(key, checked))
        mouse_click_btn.toggled.connect(lambda checked, key=gesture_key: self._on_action_type_changed(key, not checked))
        
        # Make buttons mutually exclusive
        key_press_btn.toggled.connect(lambda checked: mouse_click_btn.setChecked(not checked))
        mouse_click_btn.toggled.connect(lambda checked: key_press_btn.setChecked(not checked))
        
        action_layout.addWidget(key_press_btn)
        action_layout.addWidget(mouse_click_btn)
        action_frame.setEnabled(False)
        controls_layout.addWidget(action_frame, 2, 1)
        
        controls_layout.addWidget(QLabel("Key/Button:"), 3, 0)
        key_display = QLabel("[Space]")
        key_display.setStyleSheet("""
            background-color: #000;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 6px 12px;
            font-family: monospace;
            font-weight: bold;
        """)
        
        select_key_btn = QPushButton("Select")
        select_key_btn.setEnabled(False)
        select_key_btn.clicked.connect(lambda _, key=gesture_key: self.show_key_selection(key))
        
        key_frame = QFrame()
        key_layout = QHBoxLayout(key_frame)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(key_display)
        key_layout.addWidget(select_key_btn)
        key_layout.addStretch()
        
        controls_layout.addWidget(key_frame, 3, 1)
        
        controls_layout.addWidget(QLabel("Cooldown (s):"), 4, 0)
        cooldown_spin = QDoubleSpinBox()
        cooldown_spin.setRange(0.1, 5.0)
        cooldown_spin.setSingleStep(0.1)
        cooldown_spin.setValue(0.8)
        cooldown_spin.setEnabled(False)
        controls_layout.addWidget(cooldown_spin, 4, 1)
        
        layout.addLayout(controls_layout)
        
        self.gesture_widgets[gesture_key] = {
            'enable': enable_cb,
            'description': desc_edit,
            'key_press_btn': key_press_btn,
            'mouse_click_btn': mouse_click_btn,
            'key_display': key_display,
            'select_key_btn': select_key_btn,
            'cooldown': cooldown_spin,
            'current_key': 'space'
        }
        
        return card

    
    

    def toggle_gesture_controls(self, gesture_key, enabled):
        widgets = self.gesture_widgets[gesture_key]
        widgets['description'].setEnabled(enabled)
        widgets['key_press_btn'].parent().setEnabled(enabled)
        widgets['select_key_btn'].setEnabled(enabled)
        widgets['cooldown'].setEnabled(enabled)

    def show_key_selection(self, gesture_key):
        """Show appropriate selection dialog based on action type."""
        widgets = self.gesture_widgets[gesture_key]
        current_key = widgets['current_key']

        # Determine options based on action type
        if widgets['key_press_btn'].isChecked():
            options_list = self.valid_keyboard_keys
            dialog_title = "Select Keyboard Key"
        else:
            options_list = self.valid_mouse_buttons
            dialog_title = "Select Mouse Button"
        
        # Ensure current key is in the valid list
        if current_key not in options_list:
            current_key = options_list[0]
        
        key, ok = QInputDialog.getItem(
            self, dialog_title, 
            "Choose an option:", 
            options_list,
            options_list.index(current_key),
            False
        )
        
        if ok and key:
            self.update_key_display(gesture_key, key)

    def update_key_display(self, gesture_key, key_text):
        """Update the key display label."""
        widgets = self.gesture_widgets[gesture_key]
        widgets['current_key'] = key_text
        
        # Format display text
        if '+' in key_text:
            formatted_key = f"[{key_text.upper()}]"
        else:
            formatted_key = f"[{key_text.capitalize()}]"
        
        widgets['key_display'].setText(formatted_key)

    def get_gesture_display_name(self, gesture_key):
        gesture_names = {
            'right_index_bent': '👉 Right Hand: Bend INDEX finger down',
            'left_index_bent': '👈 Left Hand: Bend INDEX finger down', 
            'right_index_middle_bent': '👉 Right Hand: Bend INDEX + MIDDLE fingers down',
            'left_index_middle_bent': '👈 Left Hand: Bend INDEX + MIDDLE fingers down',
            'fist_gesture': '✊ Either Hand: Make a FIST (close all fingers)',
        }
        return gesture_names.get(gesture_key, gesture_key.replace('_', ' ').title())

    def load_existing_mode(self):
        """Load existing mode data into the dialog."""
        if not self.mode_data:
            return
            
        for gesture_key, gesture_data in self.mode_data.gestures.items():
            if gesture_key in self.gesture_widgets:
                widgets = self.gesture_widgets[gesture_key]
                
                # Enable the gesture
                widgets['enable'].setChecked(True)
                self.toggle_gesture_controls(gesture_key, True)
                
                # Set description
                widgets['description'].setText(gesture_data.description)
                
                # Set action type and key/button
                if gesture_data.action == 'key_press':
                    widgets['key_press_btn'].setChecked(True)
                    widgets['mouse_click_btn'].setChecked(False)
                    widgets['current_key'] = gesture_data.key or 'space'
                else:
                    widgets['key_press_btn'].setChecked(False)
                    widgets['mouse_click_btn'].setChecked(True)
                    widgets['current_key'] = gesture_data.button or 'left'
                
                # Update display
                self.update_key_display(gesture_key, widgets['current_key'])
                
                # Set cooldown
                widgets['cooldown'].setValue(gesture_data.cooldown)

    def save_mode(self):
        """Save the custom mode."""
        if self.edit_mode and self.mode_data:
            mode_name = self.mode_data.name
        else:
            mode_name = self.mode_name_edit.text().strip()
            if not mode_name:
                QMessageBox.warning(self, "Invalid Name", "Please enter a mode name!")
                return
        
        gestures = {}
        enabled_count = 0
        
        for gesture_key, widgets in self.gesture_widgets.items():
            if widgets['enable'].isChecked():
                enabled_count += 1
                
                # Determine action and key/button based on selection
                if widgets['key_press_btn'].isChecked():
                    action = 'key_press'
                    key = widgets['current_key']
                    button = None
                else:
                    action = 'mouse_click'
                    key = None
                    button = widgets['current_key']
                
                # Explicitly ensure description is always a string
                desc_from_widget = widgets['description'].text().strip()
                if desc_from_widget:
                    final_description = desc_from_widget
                else:
                    final_description = self.get_gesture_display_name(gesture_key)
                
                # Ensure final_description is definitely a string
                assert isinstance(final_description, str), "Description must be a string"
                
                gestures[gesture_key] = ApplicationModeGesture(
                    action=action,
                    key=key,
                    button=button,
                    description=final_description,
                    cooldown=widgets['cooldown'].value()
                )
        
        if enabled_count == 0:
            QMessageBox.warning(self, "No Gestures", "Please enable at least one gesture!")
            return
        
        # Create or update mode
        if self.edit_mode and self.mode_data:
            # Update existing mode
            self.mode_data.gestures = gestures
        else:
            # Create new mode
            mode_key = mode_name.lower().replace(' ', '_') + '_mode'
            mode_config = ApplicationModeConfig(
                name=mode_name,
                enabled=False,
                gestures=gestures
            )
            setattr(config_manager.app_modes, mode_key, mode_config)
        
        config_manager.save_config()
        
        action = "updated" if self.edit_mode else "created"
        QMessageBox.information(self, "Success", f"Mode '{mode_name}' {action} with {enabled_count} gestures!")
        
        self.accept()

class SettingsDialog(QDialog):
    """A dedicated dialog for all application settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Configuration Settings")
        self.setMinimumSize(800, 700)
        self.setModal(True)
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet()) # Inherit stylesheet

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Tabs for organization
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create tabs
        tabs.addTab(self.create_detection_tab(), "🎯 Detection")
        tabs.addTab(self.create_control_tab(), "🎮 Control System")
        tabs.addTab(self.create_smart_palm_tab(), "🖐️ Smart Palm")
        tabs.addTab(self.create_gesture_mapping_tab(), "🗺️ Gesture Mapping")

        # Action Buttons
        button_layout = QHBoxLayout()
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        
        self.save_btn = QPushButton("💾 Save & Apply")
        self.save_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.save_btn)
        main_layout.addLayout(button_layout)

    def _add_widget(self, layout, row, label, widget):
        """Helper to add a labeled widget to a grid layout."""
        label_widget = QLabel(label)
        layout.addWidget(label_widget, row, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignRight)

    def create_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        # Core Detection Group
        group = QGroupBox("Core Detection Parameters")
        grid = QGridLayout(group)
        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(64, 512)
        self._add_widget(grid, 0, "Input Size (px):", self.input_size_spin)

        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.1, 1.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 1, "Score Threshold:", self.score_threshold_spin)

        self.nms_threshold_spin = QDoubleSpinBox()
        self.nms_threshold_spin.setRange(0.1, 1.0)
        self.nms_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 2, "NMS Threshold:", self.nms_threshold_spin)

        self.iou_match_threshold_spin = QDoubleSpinBox()
        self.iou_match_threshold_spin.setRange(0.1, 1.0)
        self.iou_match_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 3, "IOU Match Threshold:", self.iou_match_threshold_spin)
        
        self.detection_smoothing_alpha_spin = QDoubleSpinBox()
        self.detection_smoothing_alpha_spin.setRange(0.0, 1.0)
        self.detection_smoothing_alpha_spin.setSingleStep(0.05)
        self._add_widget(grid, 4, "Detection Smoothing Alpha:", self.detection_smoothing_alpha_spin)

        self.landmark_score_threshold_spin = QDoubleSpinBox()
        self.landmark_score_threshold_spin.setRange(0.1, 1.0)
        self.landmark_score_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 5, "Landmark Redetection Threshold:", self.landmark_score_threshold_spin)

        self.always_run_palm_cb = QCheckBox("Always Run Palm Detection")
        grid.addWidget(self.always_run_palm_cb, 6, 0, 1, 2)
        
        self.enable_finger_detection_cb = QCheckBox("Enable Finger Angle Detection")
        grid.addWidget(self.enable_finger_detection_cb, 7, 0, 1, 2)
        layout.addWidget(group)

        # Visuals Group
        group_vis = QGroupBox("Visuals & Overlays")
        grid_vis = QGridLayout(group_vis)
        self.show_landmarks_cb = QCheckBox("Show Hand Landmarks")
        grid_vis.addWidget(self.show_landmarks_cb, 0, 0)
        self.show_static_gestures_cb = QCheckBox("Show Static Gestures (e.g., Fist)")
        grid_vis.addWidget(self.show_static_gestures_cb, 1, 0)
        layout.addWidget(group_vis)

        layout.addStretch()
        return tab

    def create_control_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        # Cursor Control
        group_cursor = QGroupBox("Cursor Control")
        grid_cursor = QGridLayout(group_cursor)
        self.enable_cursor_cb = QCheckBox("Enable Cursor Control")
        grid_cursor.addWidget(self.enable_cursor_cb, 0, 0, 1, 2)

        self.cursor_smoothing_spin = QDoubleSpinBox()
        self.cursor_smoothing_spin.setRange(0.0, 1.0)
        self.cursor_smoothing_spin.setSingleStep(0.05)
        self._add_widget(grid_cursor, 1, "Cursor Smoothing:", self.cursor_smoothing_spin)

        self.cursor_sensitivity_spin = QDoubleSpinBox()
        self.cursor_sensitivity_spin.setRange(0.5, 10.0)
        self.cursor_sensitivity_spin.setSingleStep(0.1)
        self._add_widget(grid_cursor, 2, "Cursor Sensitivity:", self.cursor_sensitivity_spin)
        layout.addWidget(group_cursor)

        # Scroll Control
        group_scroll = QGroupBox("Scroll Control")
        grid_scroll = QGridLayout(group_scroll)
        self.enable_scroll_cb = QCheckBox("Enable Scroll Control")
        grid_scroll.addWidget(self.enable_scroll_cb, 0, 0, 1, 2)

        self.scroll_sensitivity_spin = QSpinBox()
        self.scroll_sensitivity_spin.setRange(1, 50)
        self._add_widget(grid_scroll, 1, "Scroll Sensitivity:", self.scroll_sensitivity_spin)

        self.scroll_threshold_spin = QDoubleSpinBox()
        self.scroll_threshold_spin.setRange(0.01, 0.2)
        self.scroll_threshold_spin.setSingleStep(0.01)
        self._add_widget(grid_scroll, 2, "Scroll Activation Threshold:", self.scroll_threshold_spin)

        self.scroll_smoothing_spin = QDoubleSpinBox()
        self.scroll_smoothing_spin.setRange(0.0, 1.0)
        self.scroll_smoothing_spin.setSingleStep(0.05)
        self._add_widget(grid_scroll, 3, "Scroll Smoothing:", self.scroll_smoothing_spin)

        self.scroll_hand_pref_combo = QComboBox()
        self.scroll_hand_pref_combo.addItems(['any', 'left', 'right'])
        self._add_widget(grid_scroll, 4, "Scroll Hand Preference:", self.scroll_hand_pref_combo)
        layout.addWidget(group_scroll)

        # Keyboard Control
        group_key = QGroupBox("Keyboard Control")
        grid_key = QGridLayout(group_key)
        self.enable_key_control_cb = QCheckBox("Enable General Key Control")
        grid_key.addWidget(self.enable_key_control_cb, 0, 0, 1, 2)

        self.key_cooldown_spin = QDoubleSpinBox()
        self.key_cooldown_spin.setRange(0.1, 5.0)
        self.key_cooldown_spin.setSingleStep(0.1)
        self.key_cooldown_spin.setSuffix(" s")
        self._add_widget(grid_key, 1, "Key Press Cooldown:", self.key_cooldown_spin)
        layout.addWidget(group_key)

        layout.addStretch()
        return tab

    def create_smart_palm_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        group = QGroupBox("Smart Palm State Machine")
        grid = QGridLayout(group)
        
        self.grace_period_spin = QDoubleSpinBox()
        self.grace_period_spin.setRange(0.1, 5.0)
        self.grace_period_spin.setSingleStep(0.1)
        self.grace_period_spin.setSuffix(" s")
        self._add_widget(grid, 0, "Grace Period Duration:", self.grace_period_spin)

        self.periodic_check_spin = QSpinBox()
        self.periodic_check_spin.setRange(5, 100)
        self.periodic_check_spin.setSuffix(" frames")
        self._add_widget(grid, 1, "Periodic Check Interval:", self.periodic_check_spin)

        self.state_debug_cb = QCheckBox("Enable State Transition Debugging")
        grid.addWidget(self.state_debug_cb, 2, 0, 1, 2)
        
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def create_gesture_mapping_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        group = QGroupBox("Gesture Mapping Engine")
        grid = QGridLayout(group)

        self.enable_gesture_mapping_cb = QCheckBox("Enable Gesture Mapping System")
        grid.addWidget(self.enable_gesture_mapping_cb, 0, 0, 1, 2)

        self.require_simultaneous_cb = QCheckBox("Require Simultaneous Detection for Combos")
        grid.addWidget(self.require_simultaneous_cb, 1, 0, 1, 2)

        self.global_cooldown_spin = QDoubleSpinBox()
        self.global_cooldown_spin.setRange(0.0, 2.0)
        self.global_cooldown_spin.setSingleStep(0.05)
        self.global_cooldown_spin.setSuffix(" s")
        self._add_widget(grid, 2, "Global Cooldown:", self.global_cooldown_spin)

        self.gesture_stability_frames_spin = QSpinBox()
        self.gesture_stability_frames_spin.setRange(1, 20)
        self.gesture_stability_frames_spin.setSuffix(" frames")
        self._add_widget(grid, 3, "Gesture Stability Frames:", self.gesture_stability_frames_spin)
        layout.addWidget(group)

        group_thresh = QGroupBox("Finger Angle Thresholds")
        grid_thresh = QGridLayout(group_thresh)
        self.relaxed_thresh_spin = QSpinBox()
        self.relaxed_thresh_spin.setRange(30, 100)
        self.relaxed_thresh_spin.setSuffix("°")
        self._add_widget(grid_thresh, 0, "Relaxed Angle Max:", self.relaxed_thresh_spin)

        self.bent_thresh_spin = QSpinBox()
        self.bent_thresh_spin.setRange(100, 180)
        self.bent_thresh_spin.setSuffix("°")
        self._add_widget(grid_thresh, 1, "Bent Angle Min:", self.bent_thresh_spin)
        layout.addWidget(group_thresh)

        layout.addStretch()
        return tab

    def load_settings(self):
        """Load all settings from config manager into the UI."""
        # Detection
        self.input_size_spin.setValue(config_manager.detection.input_size)
        self.score_threshold_spin.setValue(config_manager.detection.score_threshold)
        self.nms_threshold_spin.setValue(config_manager.detection.nms_threshold)
        self.iou_match_threshold_spin.setValue(config_manager.detection.iou_match_threshold)
        self.detection_smoothing_alpha_spin.setValue(config_manager.detection.detection_smoothing_alpha)
        self.landmark_score_threshold_spin.setValue(config_manager.detection.landmark_score_for_palm_redetection_threshold)
        self.always_run_palm_cb.setChecked(config_manager.detection.always_run_palm_detection)
        self.enable_finger_detection_cb.setChecked(config_manager.detection.enable_finger_detection)
        self.show_landmarks_cb.setChecked(config_manager.detection.show_landmarks)
        self.show_static_gestures_cb.setChecked(config_manager.detection.show_static_gestures)

        # Control System
        self.enable_cursor_cb.setChecked(config_manager.control_system.enable_cursor_control)
        self.cursor_smoothing_spin.setValue(config_manager.control_system.cursor_smoothing)
        self.cursor_sensitivity_spin.setValue(config_manager.control_system.cursor_sensitivity)
        self.enable_scroll_cb.setChecked(config_manager.control_system.enable_scroll_control)
        self.scroll_sensitivity_spin.setValue(config_manager.control_system.scroll_sensitivity)
        self.scroll_threshold_spin.setValue(config_manager.control_system.scroll_threshold)
        self.scroll_smoothing_spin.setValue(config_manager.control_system.scroll_smoothing)
        self.scroll_hand_pref_combo.setCurrentText(config_manager.control_system.scroll_hand_preference)
        self.enable_key_control_cb.setChecked(config_manager.control_system.enable_key_control)
        self.key_cooldown_spin.setValue(config_manager.control_system.key_press_cooldown)

        # Smart Palm
        self.grace_period_spin.setValue(config_manager.smart_palm.grace_period_duration)
        self.periodic_check_spin.setValue(config_manager.smart_palm.periodic_check_interval)
        self.state_debug_cb.setChecked(config_manager.smart_palm.state_transition_debug)

        # Gesture Mapping
        self.enable_gesture_mapping_cb.setChecked(config_manager.gesture_mapping.enable_gesture_mapping)
        self.global_cooldown_spin.setValue(config_manager.gesture_mapping.global_cooldown)
        self.relaxed_thresh_spin.setValue(config_manager.gesture_mapping.relaxed_threshold)
        self.bent_thresh_spin.setValue(config_manager.gesture_mapping.bent_threshold)
        self.require_simultaneous_cb.setChecked(config_manager.gesture_mapping.require_simultaneous_detection)
        self.gesture_stability_frames_spin.setValue(config_manager.gesture_mapping.gesture_stability_frames)

    def save_settings(self):
        """Save all settings from UI to config manager."""
        # Detection
        config_manager.detection.input_size = self.input_size_spin.value()
        config_manager.detection.score_threshold = self.score_threshold_spin.value()
        config_manager.detection.nms_threshold = self.nms_threshold_spin.value()
        config_manager.detection.iou_match_threshold = self.iou_match_threshold_spin.value()
        config_manager.detection.detection_smoothing_alpha = self.detection_smoothing_alpha_spin.value()
        config_manager.detection.landmark_score_for_palm_redetection_threshold = self.landmark_score_threshold_spin.value()
        config_manager.detection.always_run_palm_detection = self.always_run_palm_cb.isChecked()
        config_manager.detection.enable_finger_detection = self.enable_finger_detection_cb.isChecked()
        config_manager.detection.show_landmarks = self.show_landmarks_cb.isChecked()
        config_manager.detection.show_static_gestures = self.show_static_gestures_cb.isChecked()

        # Control System
        config_manager.control_system.enable_cursor_control = self.enable_cursor_cb.isChecked()
        config_manager.control_system.cursor_smoothing = self.cursor_smoothing_spin.value()
        config_manager.control_system.cursor_sensitivity = self.cursor_sensitivity_spin.value()
        config_manager.control_system.enable_scroll_control = self.enable_scroll_cb.isChecked()
        config_manager.control_system.scroll_sensitivity = self.scroll_sensitivity_spin.value()
        config_manager.control_system.scroll_threshold = self.scroll_threshold_spin.value()
        config_manager.control_system.scroll_smoothing = self.scroll_smoothing_spin.value()
        config_manager.control_system.scroll_hand_preference = self.scroll_hand_pref_combo.currentText()
        config_manager.control_system.enable_key_control = self.enable_key_control_cb.isChecked()
        config_manager.control_system.key_press_cooldown = self.key_cooldown_spin.value()

        # Smart Palm
        config_manager.smart_palm.grace_period_duration = self.grace_period_spin.value()
        config_manager.smart_palm.periodic_check_interval = self.periodic_check_spin.value()
        config_manager.smart_palm.state_transition_debug = self.state_debug_cb.isChecked()

        # Gesture Mapping
        config_manager.gesture_mapping.enable_gesture_mapping = self.enable_gesture_mapping_cb.isChecked()
        config_manager.gesture_mapping.global_cooldown = self.global_cooldown_spin.value()
        config_manager.gesture_mapping.relaxed_threshold = self.relaxed_thresh_spin.value()
        config_manager.gesture_mapping.bent_threshold = self.bent_thresh_spin.value()
        config_manager.gesture_mapping.require_simultaneous_detection = self.require_simultaneous_cb.isChecked()
        config_manager.gesture_mapping.gesture_stability_frames = self.gesture_stability_frames_spin.value()

        config_manager.save_config()
        QMessageBox.information(self, "Settings Saved", "All settings have been applied and saved successfully!")

    def reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings", 
            "Are you sure you want to reset all settings to their defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            config_manager.reset_to_defaults()
            self.load_settings()
            # The main window needs to be refreshed too
            main_window = self.parent()
            if main_window and isinstance(main_window, GestureDashboard):
                main_window.refresh_mode_list()
                main_window.update_mode_combo()
            QMessageBox.information(self, "Settings Reset", "Settings have been reset to their default values.")

    def accept(self):
        self.save_settings()
        super().accept()

class GestureDashboard(QMainWindow):
    """
    Modern gesture control dashboard with custom mode functionality.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control System - Neon Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1200, 700)
        self.worker = None
        self.worker_thread = None
        self.gesture_widgets = {}
        self.mode_tags_group = QButtonGroup()

        self.apply_stylesheet()
        self.setup_ui()

    def apply_stylesheet(self):
        """Complete black + neon theme."""
        neon_color = "#00ffff"  # Neon Cyan
        neon_hover = "#39ff14"  # Neon Green
        bg_color = "#000000"
        border_color = "#222222"
        text_color = "#ffffff"
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg_color};
                color: {text_color};
                font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
                font-size: 12px;
                border: none;
            }}
            
            /* Video Area */
            #VideoFrame {{
                background-color: {bg_color};
                border: 2px solid {neon_color};
                border-radius: 12px;
            }}
            #VideoLabel {{
                background-color: rgba(0, 0, 0, 0.8);
                color: #888888;
                border-radius: 8px;
                font-size: 20px;
                font-weight: 600;
            }}
            
            /* Group Boxes */
            QGroupBox {{
                font-weight: 700;
                font-size: 13px;
                border: 2px solid {border_color};
                border-radius: 8px;
                margin-top: 12px;
                padding: 20px 15px 15px 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 14px;
                margin-left: 8px;
                color: {neon_color};
                background-color: {bg_color};
                border: 1px solid {neon_color};
                border-radius: 6px;
                font-weight: 700;
            }}
            
            /* Mode Tags */
            QPushButton#ModeTag {{
                background-color: transparent;
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 15px;
                padding: 8px 18px;
                font-weight: 600;
                text-align: center;
            }}
            QPushButton#ModeTag:hover {{
                border-color: {neon_hover};
                color: {neon_hover};
            }}
            QPushButton#ModeTag:checked {{
                background-color: {neon_color};
                color: {bg_color};
                border-color: {neon_color};
            }}

            /* Buttons */
            QPushButton {{
                background-color: {neon_color};
                color: {bg_color};
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-weight: 700;
                font-size: 12px;
            }}
            QPushButton:hover {{ 
                background-color: {neon_hover};
            }}
            QPushButton:pressed {{ 
                background-color: #333;
                color: {neon_hover};
            }}
            QPushButton#StartBtn {{ background-color: #00b300; }}
            QPushButton#StartBtn:hover {{ background-color: #00d900; }}
            QPushButton#StopBtn {{ background-color: #cc0000; }}
            QPushButton#StopBtn:hover {{ background-color: #ff0000; }}
            QPushButton:disabled {{ background-color: #333; color: #777; }}
            
            /* Form Controls */
            QSlider::groove:horizontal {{
                height: 4px;
                background-color: #444;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {neon_color};
                width: 20px;
                margin: -8px 0;
                border-radius: 10px;
                border: 2px solid {bg_color};
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {neon_hover};
            }}
            
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {{
                background-color: #111;
                border: 2px solid {border_color};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 500;
            }}
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
                border-color: {neon_color};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: url(./arrow.png); /* Needs an arrow icon */
            }}
            
            QCheckBox {{
                spacing: 10px;
            }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border: 2px solid {border_color};
                border-radius: 4px;
                background-color: #111;
            }}
            QCheckBox::indicator:checked {{ 
                background-color: {neon_color};
                border-color: {neon_color};
            }}
            
            QScrollArea {{ border: none; background-color: transparent; }}
            QStatusBar {{ 
                font-weight: 600;
                border-top: 1px solid {border_color};
            }}
            QDialog {{ background-color: {bg_color}; }}
        """)

    def setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left Side: Video + Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        left_panel.setFixedWidth(520)

        video_frame = self.create_video_area()
        engine_controls = self.create_engine_controls()
        
        left_layout.addWidget(video_frame)
        left_layout.addWidget(engine_controls)

        # Right Side: Gesture Modes
        right_panel = self.create_gesture_modes_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.setup_status_bar()

    def create_video_area(self):
        frame = QFrame()
        frame.setObjectName("VideoFrame")
        layout = QVBoxLayout(frame)
        self.video_label = QLabel("🎥 Gesture Engine Offline")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(500, 375)
        layout.addWidget(self.video_label)
        return frame

    def create_engine_controls(self):
        group = QGroupBox("🎛️ System Controls")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        
        self.start_btn = QPushButton("▶️ Start Engine")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.start_engine)
        
        self.stop_btn = QPushButton("⏹️ Stop Engine")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_engine)
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_resume_engine)
        
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.clicked.connect(self.open_settings_dialog)

        self.mode_combo = QComboBox()
        self.update_mode_combo()
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        
        layout.addWidget(self.start_btn, 0, 0)
        layout.addWidget(self.stop_btn, 0, 1)
        layout.addWidget(self.pause_btn, 1, 0)
        layout.addWidget(self.settings_btn, 1, 1)
        layout.addWidget(QLabel("Active Mode:"), 2, 0)
        layout.addWidget(self.mode_combo, 2, 1)
        
        return group

    def open_settings_dialog(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def update_mode_combo(self):
        current_text = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else None
        self.mode_combo.clear()
        self.mode_combo.addItem("Disabled")
        for mode_key in dir(config_manager.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_obj = getattr(config_manager.app_modes, mode_key, None)
                if mode_obj and hasattr(mode_obj, 'name'):
                    self.mode_combo.addItem(mode_obj.name)
        if current_text:
            index = self.mode_combo.findText(current_text)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)

    def create_gesture_modes_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        nav_group = QGroupBox("🎯 Gesture Modes")
        nav_layout = QVBoxLayout(nav_group)
        
        # Tags container
        self.mode_tags_widget = QWidget()
        self.mode_tags_layout = FlowLayout()
        self.mode_tags_widget.setLayout(self.mode_tags_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.mode_tags_widget)
        
        custom_mode_btn = QPushButton("➕ Create Custom Mode")
        custom_mode_btn.clicked.connect(self.create_custom_mode)
        
        nav_layout.addWidget(scroll_area)
        nav_layout.addWidget(custom_mode_btn)
        
        self.mode_content_stack = QStackedWidget()
        
        layout.addWidget(nav_group)
        layout.addWidget(self.mode_content_stack)

        self.refresh_mode_list()
        return panel

    def refresh_mode_list(self):
        # Clear layout but keep widgets
        while self.mode_tags_layout.count():
            item = self.mode_tags_layout.takeAt(0)
            if item and item.widget():
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        # Clear stack
        while self.mode_content_stack.count():
            widget = self.mode_content_stack.widget(0)
            self.mode_content_stack.removeWidget(widget)
            if widget:
                widget.setParent(None)

        self.mode_tags_group = QButtonGroup(self)
        self.mode_tags_group.setExclusive(True)

        # Add all modes
        idx = 0
        for mode_key in dir(config_manager.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_obj = getattr(config_manager.app_modes, mode_key, None)
                if mode_obj and hasattr(mode_obj, 'name'):
                    icon = "🛠️" if "custom" in mode_key.lower() else {'ppt_mode': '📊', 'media_mode': '🎵', 'browser_mode': '🌐'}.get(mode_key, '⚙️')
                    
                    tag_button = QPushButton(f"{icon} {mode_obj.name}")
                    tag_button.setObjectName("ModeTag")
                    tag_button.setCheckable(True)
                    tag_button.clicked.connect(lambda _, i=idx: self.mode_content_stack.setCurrentIndex(i))
                    
                    self.mode_tags_layout.addWidget(tag_button)
                    self.mode_tags_group.addButton(tag_button)
                    
                    page = self.create_gesture_mode_page(mode_key)
                    self.mode_content_stack.addWidget(page)
                    idx += 1
        
        if self.mode_tags_group.buttons():
            self.mode_tags_group.buttons()[0].setChecked(True)
            self.mode_content_stack.setCurrentIndex(0)

    def create_gesture_mode_page(self, mode_key):
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        layout.setSpacing(15)
        
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        
        mode_obj = getattr(config_manager.app_modes, mode_key)
        header_label = QLabel(f"Configure {mode_obj.name}")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffff;")
        
        edit_btn = QPushButton("✏️ Edit Mode")
        edit_btn.clicked.connect(lambda: self.edit_mode(mode_key))
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(edit_btn)
        layout.addWidget(header_frame)
        
        gestures_group = QGroupBox("Active Gestures")
        gestures_layout = QVBoxLayout(gestures_group)
        
        # Create scroll area for gestures
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #111;
                width: 12px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #00ffff;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #39ff14;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)
        
        # Create container widget for scrollable content
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        
        gestures = mode_obj.gestures if hasattr(mode_obj, 'gestures') else {}
        
        if not gestures:
            scroll_layout.addWidget(QLabel("No gestures configured for this mode."))
        else:
            for gesture_key, gesture_data in gestures.items():
                card = self.create_gesture_display_card(gesture_key, gesture_data)
                scroll_layout.addWidget(card)
        
        # Add stretch to push cards to top
        scroll_layout.addStretch()
        
        # Set the scroll widget and add to main layout
        scroll_area.setWidget(scroll_widget)
        gestures_layout.addWidget(scroll_area)
        
        layout.addWidget(gestures_group)
        return page_widget

    def create_gesture_display_card(self, gesture_key, gesture_data):
        # Main card container, applying 8-point grid principles
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #151515;
                border: 1px solid #2a2a2a;
                border-radius: 16px; /* Larger radius for a softer look */
            }
        """)
        card.setMinimumHeight(100) # Increased height for more vertical space

        # Main layout with increased spacing and margins
        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24) # Generous 24px padding
        layout.setSpacing(16) # 16px space between elements

        # --- Base Style for the inner text boxes ---
        element_style = """
            QLabel {{
                background-color: #242424;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                padding: 8px 16px; /* 8px vertical, 16px horizontal padding */
                min-height: 32px;
            }}
            QLabel:hover {{
                border-color: #00ffff;
            }}
        """

        # --- 1. Gesture Name Box ---
        name_text = self.get_gesture_display_name(gesture_key)
        name_label = QLabel(name_text)
        name_label.setStyleSheet(element_style + "QLabel { color: #00ffff; font-weight: bold; }")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Handle Data ---
        if hasattr(gesture_data, 'description'):
            desc_text = gesture_data.description or "No description"
            key_val = gesture_data.key or gesture_data.button
        else:
            desc_text = gesture_data.get('description', "No description")
            key_val = gesture_data.get('key') or gesture_data.get('button')

        # --- 2. Description Box ---
        desc_label = QLabel(desc_text)
        desc_label.setStyleSheet(element_style + "QLabel { color: #dddddd; }")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 3. Key Binding Badge (more prominent) ---
        formatted_key = f"[{key_val.title()}]" if key_val else "[None]"
        key_label = QLabel(formatted_key)
        key_label.setStyleSheet(element_style + """
            QLabel { 
                color: #00ffff; 
                font-family: 'Consolas', monospace; 
                font-weight: bold;
                padding: 12px 20px; /* Increased padding to make it stand out */
            }
        """)
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Add Widgets to Layout ---
        # The separator is removed for a cleaner look
        layout.addWidget(name_label, 3)
        layout.addWidget(desc_label, 5)
        layout.addStretch(1) # Flexible space
        layout.addWidget(key_label, 2)

        return card

    def get_gesture_display_name(self, gesture_key):
        gesture_names = {
            'right_index_bent': '👉 Right Index Bent',
            'left_index_bent': '👈 Left Index Bent', 
            'right_index_middle_bent': '👉 Right Index+Middle Bent',
            'left_index_middle_bent': '👈 Left Index+Middle Bent',
            'fist_gesture': '✊ Fist Gesture',
        }
        return gesture_names.get(gesture_key, gesture_key.replace('_', ' ').title())

    

    def create_custom_mode(self):
        dialog = CustomModeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_mode_list()
            self.update_mode_combo()

    def edit_mode(self, mode_key):
        mode_obj = getattr(config_manager.app_modes, mode_key)
        dialog = CustomModeDialog(self, edit_mode=mode_key, mode_data=mode_obj)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_mode_list()

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Start the engine to begin gesture control.")

    # Engine control methods
    def start_engine(self):
        if self.worker_thread is not None: return
        self.worker = GestureEngineWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.new_frame.connect(self.update_video_display)
        self.worker.status_update.connect(self.status_bar.showMessage)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.status_bar.showMessage("Engine starting...")

    def stop_engine(self):
        if self.worker: self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker, self.worker_thread = None, None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸️ Pause")
        self.video_label.setText("🎥 Gesture Engine Offline")
        self.status_bar.showMessage("Engine stopped.")

    def pause_resume_engine(self):
        if self.worker:
            from gesture_engine import complete_engine
            if complete_engine.paused:
                complete_engine.resume()
                self.pause_btn.setText("⏸️ Pause")
                self.status_bar.showMessage("Engine resumed.", 2000)
            else:
                complete_engine.pause()
                self.pause_btn.setText("▶️ Resume")
                self.status_bar.showMessage("Engine paused.", 2000)

    def change_mode(self, mode_text):
        from gesture_engine import complete_engine
        
        mode_key = "disabled"
        for key in dir(config_manager.app_modes):
            if key.endswith('_mode') and not key.startswith('_'):
                mode_obj = getattr(config_manager.app_modes, key, None)
                if mode_obj and hasattr(mode_obj, 'name') and mode_obj.name == mode_text:
                    mode_key = key
                    break
        
        if hasattr(complete_engine, 'switch_mode'):
            complete_engine.switch_mode(mode_key)
            self.status_bar.showMessage(f"Switched to {mode_text}", 2000)

    def update_video_display(self, rgb_frame):
        if rgb_frame is not None:
            h, w, ch = rgb_frame.shape
            q_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_engine()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GestureDashboard()
    window.show()
    sys.exit(app.exec())