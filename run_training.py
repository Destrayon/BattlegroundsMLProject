import time
import torch as th
import torch.nn.functional as F
from idm import InverseDynamicsModel
import json
from data_loader import VideoDataset

EPOCHS = 2
BATCH_SIZE = 8
N_WORKERS = 1
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0
LOSS_REPORT_RATE = 100

def train(model, dataloader, optimizer, device):
    model.train()
    start_time = time.time()
    loss_sum = 0
    batch_count = 0

    for epoch in range(EPOCHS):
        for batch in dataloader:
            frames = batch['frames'].to(device)
            clicks = batch['clicks'].to(device)
            mouse_positions = batch['mouse_positions'].to(device)

            batch_size, seq_length = frames.shape[:2]
            first = th.zeros((batch_size, 1), device=device)

            optimizer.zero_grad()
            batch_loss = 0

            # Initialize hidden state
            state = model.initial_state(batch_size)

            for t in range(seq_length):
                frame = frames[:, t:t+1]  # Keep time dimension
                click = clicks[:, t]
                mouse_pos = mouse_positions[:, t]

                # Forward pass
                mouse_coords, left_clicks, state = model.predict(frame, first, batch_size, state)

                # Convert numpy arrays to tensors
                mouse_coords = th.from_numpy(mouse_coords).to(device)
                left_clicks = th.from_numpy(left_clicks).float().to(device)

                # Compute loss
                click_loss = F.binary_cross_entropy(left_clicks.squeeze(), click)
                mouse_pos_loss = F.mse_loss(mouse_coords, mouse_pos)
                loss = click_loss + mouse_pos_loss

                batch_loss += loss.item()
                loss.backward()

            # Gradient clipping
            th.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            # Optimizer step
            optimizer.step()

            loss_sum += batch_loss / seq_length
            batch_count += 1

            if batch_count % LOSS_REPORT_RATE == 0:
                time_since_start = time.time() - start_time
                avg_loss = loss_sum / LOSS_REPORT_RATE
                print(f"Time: {time_since_start:.2f}, Epoch: {epoch}, Batches: {batch_count}, Avg loss: {avg_loss:.4f}")
                loss_sum = 0

    th.save(model.state_dict(), 'finetuned_weights.pth')

# Main training script
def main():
    # Load your model parameters
    with open("idm_model_parameters.json", 'r') as f:
        agent_parameters = json.load(f)

    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]

    # Initialize the model
    model = InverseDynamicsModel(        
        img_shape=tuple(net_kwargs['img_shape']),
        conv3d_params=net_kwargs['conv3d_params'],
        impala_kwargs=net_kwargs['impala_kwargs'],
        hidsize=net_kwargs['hidsize'],
        timesteps=net_kwargs['timesteps'],
        recurrence_type=net_kwargs['recurrence_type'],
        recurrence_is_residual=net_kwargs['recurrence_is_residual'],
        use_pointwise_layer=net_kwargs['use_pointwise_layer'],
        pointwise_ratio=net_kwargs['pointwise_ratio'],
        pointwise_use_activation=net_kwargs['pointwise_use_activation'],
        attention_mask_style=net_kwargs['attention_mask_style'],
        attention_heads=net_kwargs['attention_heads'],
        attention_memory_size=net_kwargs['attention_memory_size'],
        n_recurrence_layers=net_kwargs['n_recurrence_layers'],
        init_norm_kwargs=net_kwargs['init_norm_kwargs']
    ).to(DEVICE)
    
    dataset = VideoDataset("output")
    dataloader = th.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(model, dataloader, optimizer, DEVICE)

if __name__ == "__main__":
    main()