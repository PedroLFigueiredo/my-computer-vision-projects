import cv2
import mediapipe as mp

def initialize_hand_tracking():
    mp_hand = mp.solutions.hands
    hand = mp_hand.Hands()
    mp_draw = mp.solutions.drawing_utils
    connections_style = mp_draw.DrawingSpec(color=(0, 255, 0))
    return hand, mp_draw, mp_hand, connections_style

def process_frame(frame, hand_tracker, mp_draw, mp_hand, connections_style):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hand.HAND_CONNECTIONS, connection_drawing_spec=connections_style)
    
    return frame

def capture_video():
    video = cv2.VideoCapture(0)
    hand_tracker, mp_draw, mp_hand, connections_style = initialize_hand_tracking()

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame = process_frame(frame, hand_tracker, mp_draw, mp_hand, connections_style)
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
