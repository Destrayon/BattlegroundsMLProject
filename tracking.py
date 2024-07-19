import obspython as obs
import json
import os
from pynput import keyboard, mouse
from ctypes import windll, Structure, c_long, byref

print("Script is being loaded...")

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

class FrameInputTracker:
    def __init__(self):
        self.tracking = False
        self.frame_data = []
        self.current_frame = {}
        self.keys_pressed = set()
        self.mouse_buttons = {'left': False, 'right': False}
        self.mouse_position = (0, 0)
        self.frame_count = -1
        self.output_width = 1920  # Default value
        self.output_height = 1080  # Default value
        self.current_output = None
        
        # Set up listeners
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.mouse_listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def on_key_press(self, key):
        if self.tracking:
            self.keys_pressed.add(self.key_to_string(key))

    def on_key_release(self, key):
        if self.tracking:
            key_str = self.key_to_string(key)
            self.keys_pressed.discard(key_str)

    def key_to_string(self, key):
        try:
            return key.char
        except AttributeError:
            return str(key)

    def on_click(self, x, y, button, pressed):
        if self.tracking:
            if button == mouse.Button.left:
                self.mouse_buttons['left'] = pressed
            elif button == mouse.Button.right:
                self.mouse_buttons['right'] = pressed

    def on_move(self, x, y):
        if self.tracking:
            # Get the position relative to the primary monitor
            point = POINT()
            windll.user32.GetCursorPos(byref(point))
            self.mouse_position = (point.x, point.y)

    def get_primary_screen_width(self):
        return windll.user32.GetSystemMetrics(0)

    def get_primary_screen_height(self):
        return windll.user32.GetSystemMetrics(1)

    def start_tracking(self):
        self.tracking = True
        self.frame_data = []
        self.frame_count = 0
        self.current_output = obs.obs_frontend_get_recording_output()
        if self.current_output:
            self.output_width = obs.obs_output_get_width(self.current_output)
            self.output_height = obs.obs_output_get_height(self.current_output)
        print(f"DEBUG: Started tracking. Output resolution: {self.output_width}x{self.output_height}")

    def stop_tracking(self):
        self.tracking = False
        self.save_to_file()
        if self.current_output:
            obs.obs_output_release(self.current_output)
            self.current_output = None

    def update_frame(self):
        if self.tracking:
            self.current_frame = {
                'frame': self.frame_count,
                'keys_pressed': list(self.keys_pressed),
                'mouse_buttons': self.mouse_buttons.copy(),
                'mouse_position': self.mouse_position
            }
            self.frame_data.append(self.current_frame)
            self.frame_count += 1

    def save_to_file(self):
        video_path = obs.obs_frontend_get_last_recording()
        if not video_path:
            print("Error: Could not get the path of the last recording.")
            return

        print(f"DEBUG: Last recording path: {video_path}")
        print(f"DEBUG: Video dimensions: {self.output_width}x{self.output_height}")

        # Remove the first frame (frame -1) and adjust frame numbers
        adjusted_frame_data = self.frame_data[1:]
        for i, frame in enumerate(adjusted_frame_data):
            frame['frame'] = i

        # Scale and clamp mouse positions
        for frame in adjusted_frame_data:
            x, y = frame['mouse_position']
            scaled_x = x * self.output_width / self.get_primary_screen_width()
            scaled_y = y * self.output_height / self.get_primary_screen_height()
            clamped_x = clamp(scaled_x, 0, self.output_width - 1)
            clamped_y = clamp(scaled_y, 0, self.output_height - 1)
            frame['mouse_position'] = (clamped_x, clamped_y)
        
        # Generate the input data filename based on the video filename
        directory = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        input_data_path = os.path.join(directory, f"{base_name}_input.json")
        
        print(f"DEBUG: Saving input data to: {input_data_path}")

        try:
            with open(input_data_path, 'w') as f:
                json.dump(adjusted_frame_data, f)
            print(f"Frame-by-frame input data saved to {input_data_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")

tracker = FrameInputTracker()

def script_update(settings):
    print("Script updated")

def script_description():
    return "Tracks frame-by-frame keyboard and mouse inputs during recording."

def script_properties():
    props = obs.obs_properties_create()
    return props

def on_event(event):
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED:
        print("Recording started")
        tracker.start_tracking()
    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED:
        print("Recording stopped")
        tracker.stop_tracking()

def script_load(settings):
    print("Script loaded")
    obs.obs_frontend_add_event_callback(on_event)

def script_tick(seconds):
    tracker.update_frame()

print("Script defined successfully")