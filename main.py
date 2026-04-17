import cv2
import argparse
from detector import DrowsinessDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    detector = DrowsinessDetector()

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting Drowsiness Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for selfie view
        frame = cv2.flip(frame, 1)

        # Process frame
        processed_frame = detector.process_frame(frame)

        # Display resulting frame
        cv2.imshow('Drowsiness Detection', processed_frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

if __name__ == '__main__':
    main()
