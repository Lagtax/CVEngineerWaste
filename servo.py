import torch
from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time
from threading import Lock, Timer

# Check and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# GPIO Setup
GPIO.setmode(GPIO.BOARD)
SERVO1_PIN = 11  # For Inorganic
SERVO2_PIN = 12  # For Organic
GPIO.setup(SERVO1_PIN, GPIO.OUT)
GPIO.setup(SERVO2_PIN, GPIO.OUT)

# Create PWM objects for servos
servo1 = GPIO.PWM(SERVO1_PIN, 50)  # 50Hz frequency
servo2 = GPIO.PWM(SERVO2_PIN, 50)
servo1.start(0)
servo2.start(0)

# Create locks for servo control
servo1_lock = Lock()
servo2_lock = Lock()

# Load the model
model = YOLO("Trash Recog.v2i.yolov8.pt")
model.to(device)


def set_servo_angle(servo, angle):
    """Convert angle to duty cycle and set servo position"""
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)


def servo_sequence(servo_num, lock):
    """Execute the servo motion sequence"""
    servo = servo1 if servo_num == 1 else servo2

    with lock:
        # Rotate to 90 degrees
        set_servo_angle(servo, 90)
        time.sleep(4)
        # Return to 0 degrees
        set_servo_angle(servo, 0)
        time.sleep(0.5)  # Small delay to ensure movement completion


def predict_and_sort():
    results = model.predict(
        source="0",
        show=True,
        conf=0.3,
        iou=0.5,
        imgsz=320,
        stream=True,
        verbose=False,
        max_det=3
    )

    for result in results:
        boxes = result.boxes

        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                print(f"Detected: {label} (Confidence: {conf:.2f})")

                # Handle Inorganic detection
                if label == "Inorganic" and not servo1_lock.locked():
                    print("Moving servo 1 for Inorganic waste")
                    Timer(0, servo_sequence, args=(1, servo1_lock)).start()

                # Handle Organic detection
                elif label == "Organic" and not servo2_lock.locked():
                    print("Moving servo 2 for Organic waste")
                    Timer(0, servo_sequence, args=(2, servo2_lock)).start()
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break


def cleanup():
    """Clean up GPIO on program exit"""
    cv2.destroyAllWindows()
    servo1.stop()
    servo2.stop()
    GPIO.cleanup()


if __name__ == "__main__":
    try:
        predict_and_sort()
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        cleanup()