import cv2
import numpy as np
import openvino as ov
from pathlib import Path
from typing import Optional, Tuple
import threading

class ModelManager:
    """Manages OpenVINO models for gesture detection"""
    
    def __init__(self):
        self.core = ov.Core()
        self.models = {}
        self.compiled_models = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize_models(self, model_paths: dict) -> bool:
        """Initialize all required models"""
        with self._lock:
            if self._initialized:
                return True
            
            try:
                # Load palm detection model
                if 'palm_detection' in model_paths:
                    self._load_palm_detection_model(model_paths['palm_detection'])
                
                # Load landmark model
                if 'hand_landmarks' in model_paths:
                    self._load_landmark_model(model_paths['hand_landmarks'])
                
                # Load gesture models
                if 'gesture_embedder' in model_paths:
                    self._load_gesture_embedder(model_paths['gesture_embedder'])
                
                if 'gesture_classifier' in model_paths:
                    self._load_gesture_classifier(model_paths['gesture_classifier'])
                
                self._initialized = True
                print("✅ All models initialized successfully")
                return True
                
            except Exception as e:
                print(f"❌ Model initialization failed: {e}")
                return False
    
    def _load_palm_detection_model(self, model_path: str):
        """Load and compile palm detection model"""
        from openvino.preprocess import PrePostProcessor, ColorFormat
        from openvino import Type, Layout
        
        model = self.core.read_model(model_path)
        
        # Apply preprocessing
        ppp_pd = PrePostProcessor(model)
        ppp_pd.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
        
        ppp_pd.input().model().set_layout(Layout('NHWC'))
        ppp_pd.input().preprocess() \
            .convert_element_type(Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .scale([255.0, 255.0, 255.0])
        
        palm_detection_model = ppp_pd.build()
        compiled_model = self.core.compile_model(palm_detection_model, "CPU")
        
        self.models['palm_detection'] = palm_detection_model
        self.compiled_models['palm_detection'] = compiled_model
    
    def _load_landmark_model(self, model_path: str):
        """Load landmark detection model"""
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, "CPU")
        
        self.models['hand_landmarks'] = model
        self.compiled_models['hand_landmarks'] = compiled_model
    
    def _load_gesture_embedder(self, model_path: str):
        """Load gesture embedding model"""
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, "CPU")
        
        self.models['gesture_embedder'] = model
        self.compiled_models['gesture_embedder'] = compiled_model
    
    def _load_gesture_classifier(self, model_path: str):
        """Load gesture classification model"""
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, "CPU")
        
        self.models['gesture_classifier'] = model
        self.compiled_models['gesture_classifier'] = compiled_model
    
    def get_compiled_model(self, model_name: str):
        """Get compiled model by name"""
        with self._lock:
            return self.compiled_models.get(model_name)
    
    def is_initialized(self) -> bool:
        """Check if models are initialized"""
        return self._initialized
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        with self._lock:
            return {
                'initialized': self._initialized,
                'available_models': list(self.compiled_models.keys()),
                'model_count': len(self.compiled_models)
            }

# Global model manager instance
model_manager = ModelManager()