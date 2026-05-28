# transformer.py
import torch
import torch.nn as nn
import copy
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with configurable input/output sizes, layers,
    weight initialization, and reward computation capabilities.
    """

    def __init__(self, view_dist, output_dim=2, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, dropout=0.1, batch_first=True):
        """
        Initialize the TransformerEncoder.

        Args:
            input_dim: Dimension of the raw input features
            output_dim: Dimension of the final model output
            d_model: Dimension of the internal model embeddings
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of the feed-forward network
            dropout: Dropout probability
            batch_first: If True, input and output shapes are (batch, seq, feature)
        """
        super(TransformerEncoder, self).__init__()

        self.view_dist = view_dist
        self.input_dim = (2 * int(view_dist) + 1) ** 2
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.reward_fn = None

        # 1. Custom Input Projection
        self.input_projection = nn.Linear(self.input_dim, d_model)

        # Build encoder layer by layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first  # Set according to training needs
        )

        # Stack layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # 2. Custom Output Projection
        self.output_projection = nn.Linear(d_model, output_dim)

        # Apply default weight initialization (includes new projections)
        self._default_weight_init()

        # Set default reward function
        self._default_reward_fn = lambda x: torch.tensor(0.0)

    def step(self, agent):
        pass

    def _default_weight_init(self):
        """
        Default weight initialization using PyTorch conventions.
        Xavier uniform for linear weights, zeros for biases.
        """
        # Initialize projections
        for projection in [self.input_projection, self.output_projection]:
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

        # Initialize hidden core encoder structural layers
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                        nn.init.xavier_uniform_(module.in_proj_weight)
                    if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass through the encoder layers.

        Args:
            src: Input tensor of shape:
                 - (batch, seq_len, input_dim) if batch_first=True
                 - (seq_len, batch, input_dim) if batch_first=False
            mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            Output tensor of shape:
                 - (batch, seq_len, output_dim) if batch_first=True
                 - (seq_len, batch, output_dim) if batch_first=False
        """
        # Handle cases where an unbatched flat sequence vector or standard batch is given
        has_seq_dim = len(src.shape) == 3

        # Project input features up to structural model depth
        output = self.input_projection(src)

        # Core attention stacks tracking
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)

        # Project down to physical trajectory target space
        output = self.output_projection(output)

        # If tracking sequence vectors, extract latest frame state target
        if has_seq_dim:
            output = output[:, -1, :]

        return output

    def __call__(self, local_observation):
        """
        Interface method designed to directly catch raw NumPy patches sent
        from environment.py's step_simulation function.

        Args:
            local_observation (np.ndarray): 2D patch from the environment simulation.

        Returns:
            list: [target_x, target_y] trajectory commands
        """
        # Ensure model is in inference mode
        self.eval()

        with torch.no_grad():
            # 1. Flatten the 2D local patch observation into a 1D vector arrays
            flat_obs = local_observation.flatten()

            # 2. Convert to a PyTorch float tensor and inject a mock batch dimension [Batch=1, Input_Dim]
            tensor_input = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

            # 3. Predict targets
            prediction = self.forward(tensor_input)

            # 4. Strip batch structure and map target outputs to clean float primitives
            coordinates = prediction.squeeze(0).tolist()

        return coordinates[0], coordinates[1]

    def apply_weight_init(self, init_fn):
        """Apply custom weight initialization to all components."""
        init_fn(self.input_projection)
        init_fn(self.output_projection)
        for layer in self.layers:
            for module in layer.modules():
                if len(list(module.children())) == 0:
                    init_fn(module)

    def set_reward_fn(self, reward_fn):
        self.reward_fn = reward_fn

    def compute_reward(self, output):
        if self.reward_fn is None:
            return self._default_reward_fn(output)
        return self.reward_fn(output)