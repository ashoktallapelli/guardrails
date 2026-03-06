"""
faces.py - Face detection and blur check.

Uses OpenCV Haar Cascade for face detection and Gaussian blur for anonymization.
"""

import logging
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result

logger = logging.getLogger(__name__)


class FacesCheck(BaseCheck):
    """Detects and blurs faces in images for privacy."""

    name = "faces"
    input_type = "image"
    can_reject = False  # Faces don't reject, only redact
    can_redact = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blur_kernel_size = config.get("face_blur_kernel_size", 51)

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Detect faces in image using OpenCV Haar Cascade.

        Args:
            input_data: PIL Image
            config: Configuration

        Returns:
            CheckResult with face count and locations
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        fail_closed = config.get("fail_closed", False)

        try:
            import cv2
            import numpy as np
        except ImportError:
            return fail_result(self.name, "opencv-python not installed", fail_closed)

        try:
            # Convert PIL to OpenCV format
            cv_img = cv2.cvtColor(np.array(input_data), cv2.COLOR_RGB2BGR)

            # Load face detection cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            # Detect faces
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            face_boxes = []
            for (x, y, w, h) in faces:
                face_boxes.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })

            face_count = len(faces)
            logger.info(f"Face detection: found {face_count} face(s)")

            action = "redact" if face_count > 0 else "allow"

            return CheckResult(
                safe=True,  # Faces don't make image "unsafe", just need blurring
                score=float(face_count),
                action=action,
                details={
                    "face_count": face_count,
                    "faces": face_boxes
                }
            )

        except Exception as e:
            return fail_result(self.name, str(e), fail_closed)

    def redact(self, input_data, config: Dict[str, Any]):
        """
        Blur detected faces in image.

        Args:
            input_data: PIL Image
            config: Configuration with face_blur_kernel_size

        Returns:
            Image with blurred faces
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.warning("opencv-python not installed, skipping face blur")
            return input_data

        try:
            # Convert PIL to OpenCV format
            cv_img = cv2.cvtColor(np.array(input_data), cv2.COLOR_RGB2BGR)

            # Load face detection cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            # Detect faces
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Blur each detected face
            kernel_size = config.get("face_blur_kernel_size", self.blur_kernel_size)
            for (x, y, w, h) in faces:
                roi = cv_img[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
                cv_img[y:y+h, x:x+w] = roi

            logger.info(f"Blurred {len(faces)} face(s)")

            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        except Exception as e:
            logger.warning(f"Face blur failed: {e}")
            return input_data

    def get_reason(self, result: CheckResult) -> str:
        """Generate human-readable reason."""
        face_count = result.details.get("face_count", 0)
        if face_count > 0:
            return f"Faces blurred: {face_count}"
        return "No faces detected"
