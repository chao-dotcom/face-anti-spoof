"""
Enhanced demo for face anti-spoofing.

Supports webcam, video files, and single images with advanced features.
"""

import cv2
import numpy as np
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import FaceAntiSpoof, VideoAntiSpoof
from src.detection import FaceDetector
from src.core.exceptions import AntiSpoofError

# Model paths
MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector_quantized.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_quantized.onnx"


class DemoConfig:
    """Configuration for demo."""

    def __init__(self):
        self.show_fps = True
        self.show_system_info = True
        self.show_confidence = True
        self.temporal_filtering = True
        self.quality_check = True
        self.save_detections = False
        self.output_dir = Path("output")


def draw_detection(
    frame: np.ndarray,
    bbox: Dict[str, float],
    result: Dict[str, Any],
    config: DemoConfig,
) -> np.ndarray:
    """
    Draw detection results on frame.

    Args:
        frame: Image to draw on
        bbox: Bounding box dict
        result: Prediction result
        config: Demo configuration

    Returns:
        Frame with visualizations
    """
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])

    # Color based on result
    if result.get("ready", True):  # For temporal, check if ready
        is_real = result.get("is_real", False)
        color = (0, 255, 0) if is_real else (0, 0, 255)  # Green for real, red for spoof
        label = result.get("status", "unknown").upper()
    else:
        color = (255, 255, 0)  # Yellow for not ready
        label = "ANALYZING..."

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(
        frame,
        (x, y - label_size[1] - 10),
        (x + label_size[0] + 10, y),
        color,
        -1,
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Show confidence
    if config.show_confidence and result.get("ready", True):
        confidence = result.get("confidence", 0.0)
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(
            frame,
            conf_text,
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Show consistency for temporal
        if "consistency" in result:
            cons_text = f"Cons: {result['consistency']:.2%}"
            cv2.putText(
                frame,
                cons_text,
                (x, y + h + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    return frame


def draw_info_overlay(
    frame: np.ndarray,
    fps: float,
    config: DemoConfig,
) -> np.ndarray:
    """Draw FPS and system info overlay."""
    if not config.show_fps:
        return frame

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Help text
    help_text = [
        "Q: Quit",
        "S: Screenshot",
        "T: Toggle Temporal",
        "D: Toggle Debug",
    ]

    y_offset = frame.shape[0] - 20
    for i, text in enumerate(reversed(help_text)):
        cv2.putText(
            frame,
            text,
            (10, y_offset - i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


def process_image(
    image_path: str,
    detector: FaceDetector,
    model: FaceAntiSpoof,
    visualize: bool = True,
) -> None:
    """Process single image."""
    print(f"Processing image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect(image_rgb)
    print(f"Found {len(detections)} face(s)")

    if not detections:
        print("No faces detected")
        return

    # Process each face
    for i, detection in enumerate(detections):
        print(f"\nFace {i + 1}:")
        print(f"  Bbox: {detection['bbox']}")
        print(f"  Confidence: {detection['confidence']:.3f}")

        # Crop face
        face_crop = detector.crop_face(image_rgb, detection, padding=1.5)

        # Predict
        result = model.predict(face_crop)

        print(f"  Status: {result['status']}")
        print(f"  Is Real: {result['is_real']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Logit Diff: {result['logit_diff']:.3f}")

        # Visualize
        if visualize:
            config = DemoConfig()
            image_rgb = draw_detection(image_rgb, detection["bbox"], result, config)

    if visualize:
        # Convert back to BGR for display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Liveness", image_bgr)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    source: int | str,
    detector: FaceDetector,
    model: VideoAntiSpoof,
    config: DemoConfig,
) -> None:
    """Process video stream (webcam or file)."""
    print(f"Opening video source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # FPS tracking
    fps_history = []
    frame_time = time.time()

    print("\nStarting video processing...")
    print("Press 'Q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        frame_count += 1

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        try:
            detections = detector.detect(frame_rgb)

            # Process first face (can be extended for multi-face)
            if detections:
                detection = detections[0]

                # Crop face
                face_crop = detector.crop_face(frame_rgb, detection, padding=1.5)

                # Predict (temporal if enabled)
                if config.temporal_filtering:
                    result = model.predict_temporal(face_crop)
                else:
                    result = model.predict(face_crop)
                    result["ready"] = True  # Always ready for single-frame

                # Draw results
                frame_rgb = draw_detection(
                    frame_rgb, detection["bbox"], result, config
                )

        except AntiSpoofError as e:
            print(f"Error processing frame: {e}")

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - frame_time) if current_time > frame_time else 0
        frame_time = current_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)

        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # Draw overlay
        frame_rgb = draw_info_overlay(frame_rgb, avg_fps, config)

        # Convert back to BGR and display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Liveness", frame_bgr)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("\nQuitting...")
            break
        elif key == ord("s") or key == ord("S"):
            # Save screenshot
            if config.save_detections:
                output_path = config.output_dir / f"detection_{frame_count:06d}.jpg"
                config.output_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(output_path), frame_bgr)
                print(f"Saved screenshot: {output_path}")
        elif key == ord("t") or key == ord("T"):
            # Toggle temporal filtering
            config.temporal_filtering = not config.temporal_filtering
            print(f"Temporal filtering: {'ON' if config.temporal_filtering else 'OFF'}")
            model.reset()
        elif key == ord("d") or key == ord("D"):
            # Toggle debug info
            config.show_confidence = not config.show_confidence
            print(f"Debug info: {'ON' if config.show_confidence else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    # Print statistics
    print("\n" + "=" * 50)
    print("Session Statistics:")
    stats = model.get_stats()
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Real: {stats['real_predictions']}")
    print(f"Spoof: {stats['spoof_predictions']}")
    print(f"Spoof Rate: {stats['spoof_rate']:.2%}")
    print("=" * 50)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Face Liveness Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (default: 0 if no other source specified)",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default=str(DETECTOR_MODEL),
        help=f"Path to face detector model (default: {DETECTOR_MODEL})",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(LIVENESS_MODEL),
        help=f"Path to anti-spoofing model (default: {LIVENESS_MODEL})",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default: 0.5)",
    )

    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal filtering for video",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization for image mode",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detection screenshots (press 'S')",
    )

    args = parser.parse_args()

    # Determine source
    if args.image:
        source_type = "image"
        source = args.image
    elif args.video:
        source_type = "video"
        source = args.video
    else:
        source_type = "camera"
        source = args.camera if args.camera is not None else 0

    print("=" * 60)
    print("Face Liveness Demo")
    print("=" * 60)
    print(f"Source Type: {source_type}")
    print(f"Source: {source}")
    print(f"Detector: {args.detector}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    # Load models
    try:
        print("\nLoading face detector...")
        detector = FaceDetector(model_path=args.detector)

        if source_type == "image":
            print("Loading anti-spoof model...")
            model = FaceAntiSpoof(
                model_path=args.model,
                threshold=args.threshold,
            )

            process_image(source, detector, model, visualize=args.visualize)

        else:
            print("Loading video anti-spoof model...")
            model = VideoAntiSpoof(
                model_path=args.model,
                threshold=args.threshold,
                temporal_window=5,
                consistency_threshold=0.8,
            )

            config = DemoConfig()
            config.temporal_filtering = not args.no_temporal
            config.save_detections = args.save

            process_video(source, detector, model, config)

    except AntiSpoofError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
