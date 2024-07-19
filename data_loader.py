import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, folder_path, sequence_length=128, frame_size=(128, 128)):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_name)
        json_path = os.path.join(self.folder_path, video_name.replace('.mp4', '_input.json'))
        
        # Load video frames and metadata
        frames, metadata = self._load_frames(video_path)
        
        # Load and process JSON data
        clicks, mouse_positions = self._load_json_data(json_path, metadata)
        
        return {
            'frames': frames,
            'clicks': clicks,
            'mouse_positions': mouse_positions
        }
    
    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Get video metadata
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        
        # Randomly select a starting point
        max_start_frame = max(0, frame_count - self.sequence_length)
        start_frame = np.random.randint(0, max_start_frame + 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()
        
        # Pad if necessary
        if len(frames) < self.sequence_length:
            frames = self._pad_sequence(frames, self.sequence_length)
        
        # Normalize and convert to tensor
        frames = np.array(frames) / 255.0
        frames_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        
        metadata = {'frame_count': frame_count, 'fps': fps, 'start_frame': start_frame}
        
        return frames_tensor, metadata
    
    def _load_json_data(self, json_path, video_metadata):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Calculate the frame step based on JSON data length and video frame count
        frame_step = len(data) / video_metadata['frame_count']
        
        start_frame = video_metadata['start_frame']
        end_frame = start_frame + self.sequence_length
        
        clicks = []
        mouse_positions = []
        for i in range(start_frame, end_frame):
            json_index = min(int(i * frame_step), len(data) - 1)
            frame_data = data[json_index]
            clicks.append(int(frame_data['mouse_buttons']['left']))
            mouse_positions.append(np.array(frame_data['mouse_position']) / 128.0)
        
        # Pad if necessary
        clicks = self._pad_sequence(clicks, self.sequence_length)
        mouse_positions = self._pad_sequence(mouse_positions, self.sequence_length)
        
        return torch.FloatTensor(clicks), torch.FloatTensor(mouse_positions)
    
    def _pad_sequence(self, sequence, target_length):
        if len(sequence) < target_length:
            padding = [sequence[-1]] * (target_length - len(sequence))
            return sequence + padding
        return sequence[:target_length]

def get_dataloader(folder_path, batch_size=1, shuffle=True, num_workers=0):
    dataset = VideoDataset(folder_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

