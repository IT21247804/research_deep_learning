# sinusitis_detection_and_severity_classification/__init__.py

from .foreign_body_detection import detect_objects,run_inference

__all__ = [ 
    "detect_objects",
    "run_inference"
]
