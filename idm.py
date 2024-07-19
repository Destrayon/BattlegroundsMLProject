from lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from lib.impala_cnn import ImpalaCNN
from lib.policy import ImgPreprocessing, ImgObsProcess
from torch import nn
import torch
from copy import deepcopy
import json
from torch.nn import functional as F

class InverseDynamicsModel(nn.Module):
    def __init__(self,
                 img_shape=(3, 128, 128),
                 conv3d_params=dict(inchan=3, outchan=128, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
                 impala_chans=(64, 128, 128),
                 impala_kwargs={},
                 hidsize=512,
                 timesteps=128,
                 recurrence_type="transformer",
                 recurrence_is_residual=True,
                 use_pointwise_layer=True,
                 pointwise_ratio=4,
                 pointwise_use_activation=False,
                 attention_mask_style="none",
                 attention_heads=32,
                 attention_memory_size=128,
                 n_recurrence_layers=2,
                 init_norm_kwargs={}):
        super().__init__()
        
        self.img_preprocess = ImgPreprocessing(img_statistics=None, scale_img=True)
        
        # 3D Convolution layer
        self.conv3d = FanInInitReLULayer(
            inchan=conv3d_params['inchan'],
            outchan=conv3d_params['outchan'],
            layer_type="conv3d",
            kernel_size=conv3d_params['kernel_size'],
            padding=conv3d_params['padding'],
            **init_norm_kwargs
        )

        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        self.img_processing = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=impala_chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs
        )

        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.mouse_coord = nn.Linear(hidsize, 2)  # For x and y coordinates
        self.left_click = nn.Linear(hidsize, 1)   # For left click state
    
    def forward(self, x, first, state_in):
        x = self.img_preprocess(x)
        
        # Reshape for 3D convolution
        b, t, c, h, w = x.shape
        x = x.transpose(1, 2)  # b, c, t, h, w
        
        x = self.conv3d(x)
        
        x = self.img_processing(x)

        x, state_out = self.recurrent_layer(x, first, state_in)

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        mouse_coord = torch.clamp(self.mouse_coord(x), 0, 1)
        left_click = torch.sigmoid(self.left_click(x))
        
        mouse_coords = mouse_coord.cpu().numpy()  # Shape: (batch_size, 2)
        left_clicks = (left_click.cpu().numpy() > 0.5).astype(bool)  # Shape: (batch_size, 1)
        
        return mouse_coords, left_clicks, state_out
    
    def initial_state(self, batch_size):
        return self.recurrent_layer.initial_state(batch_size)
    
    def predict(self, input, first, batch_size, state_in = None):
        if state_in is None:
            state_in = self.initial_state(batch_size)

        return self(input, first, state_in)
    

def test_improved_idm():
    with open("idm_model_parameters.json", 'r') as f:
        agent_parameters = json.load(f)

    # Access the nested dictionary items
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
    )
    
    # Create dummy input data
    # Assuming input shape is (batch_size, time_steps, channels, height, width)
    batch_size = 2
    time_steps = 128
    channels = 3
    height = 128
    width = 128
    
    dummy_input = torch.randn(batch_size, time_steps, channels, height, width)
    
    # Move model and input to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = torch.zeros((dummy_input.shape[0], 1)).to(device)
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        mouse_coords, left_click, state_out = model.predict(dummy_input, first, batch_size)
    
    # Print output shape
    print(f"Input shape: {dummy_input.shape}")

if __name__ == "__main__":
    test_improved_idm()