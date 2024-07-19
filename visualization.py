import json
import cv2
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def annotate_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, 'input.mp4')
    data_path = os.path.join(script_dir, 'input.json')
    output_path = os.path.join(script_dir, 'output_annotated.mp4')

    try:
        input_data = load_data(data_path)
        print(f"Loaded {len(input_data)} frames of input data")
    except FileNotFoundError:
        print(f"Error: Could not find input data file: {data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in file: {data_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height} at {fps} fps, {total_frames} total frames")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_path}")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_number = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number < len(input_data):
                frame_data = input_data[frame_number]
                mouse_x, mouse_y = frame_data['mouse_position']
                left_click = frame_data['mouse_buttons']['left']
                right_click = frame_data['mouse_buttons']['right']

                if left_click and right_click:
                    color = (255, 0, 255)
                elif left_click:
                    color = (0, 255, 0)
                elif right_click:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                cv2.circle(frame, (int(mouse_x), int(mouse_y)), 10, color, -1)

                keys_pressed = frame_data['keys_pressed']
                key_text = ', '.join(keys_pressed)
                cv2.putText(frame, f"Frame: {frame_number}, Keys: {key_text}", (10, height - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(frame)
            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    except Exception as e:
        print(f"An error occurred while processing the video: {str(e)}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Video processing completed. Output saved to: {output_path}")
    print(f"Processed {frame_number} frames in total")

if __name__ == "__main__":
    annotate_video()