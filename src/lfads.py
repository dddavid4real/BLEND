# Author: Zhengrui Guo
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from src.mask import UNMASKED_LABEL
from src.model import PositionalEncoding
import math

class LFADS(nn.Module):
    def __init__(self, config, trial_length, num_neurons, device, max_spikes, behavior_dim):
        super().__init__()
        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons
        self.device = device
        self.max_spikes = max_spikes
        self.behavior_dim = behavior_dim

        # Define dimensions
        self.hidden_size = 64
        self.factor_size = 32

        # Encoder
        self.encoder = Encoder(
            input_size=num_neurons,
            hidden_size=self.hidden_size,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )

        # Controller
        self.controller = Controller(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            dropout=config.DROPOUT
        )

        # Generator
        self.generator = Generator(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=config.DROPOUT
        )

        # Factor
        self.factor = Factor(
            input_size=self.hidden_size,
            output_size=self.factor_size,
            dropout=config.DROPOUT
        )

        # Decoder
        self.decoder = Decoder(
            input_size=self.factor_size,
            hidden_size=self.hidden_size,
            output_size=num_neurons,
            dropout=config.DROPOUT
        )

        if config.LOSS.TYPE == "poisson":
            self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.LOGRATE)
        elif config.LOSS.TYPE == "cel":
            self.classifier = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {config.LOSS.TYPE}")

    def forward(self, src, mask_labels, **kwargs):
        src = src.float()
        src = src.permute(1, 0, 2)  # t x b x n

        # Encoder
        encoder_output = self.encoder(src)

        # Controller
        controller_output = self.controller(encoder_output)

        # Generator
        generator_output = self.generator(controller_output)

        # Factor
        factors = self.factor(generator_output)

        # Decoder
        output = self.decoder(factors)

        if self.config.LOSS.TYPE == "poisson":
            pred_rates = output.permute(1, 0, 2)  # b x t x n
            loss = self.classifier(pred_rates, mask_labels)
        elif self.config.LOSS.TYPE == "cel":
            spike_logits = torch.stack(torch.split(output, self.num_neurons, dim=-1), dim=-2)
            loss = self.classifier(spike_logits.permute(1, 2, 0, 3), mask_labels)

        masked_loss = loss[mask_labels != -100]  # Assuming -100 is the UNMASKED_LABEL

        if self.config.LOSS.TOPK < 1:
            topk, indices = torch.topk(masked_loss, int(len(masked_loss) * self.config.LOSS.TOPK))
            topk_mask = torch.zeros_like(masked_loss)
            masked_loss = topk_mask.scatter(0, indices, topk)

        masked_loss = masked_loss.mean()

        return masked_loss.unsqueeze(0), pred_rates if self.config.LOSS.TYPE == "poisson" else spike_logits.permute(1, 2, 0, 3)
        #     None,  # layer_outputs (not applicable for LFADS)
        #     None,  # layer_weights (not applicable for LFADS)

    def get_factor_size(self):
        return self.factor_size

    def get_hidden_size(self):
        return self.hidden_size

class LFADS_Teacher(nn.Module):
    def __init__(self, config, trial_length, num_neurons, device, max_spikes, behavior_dim):
        super().__init__()
        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons
        self.device = device
        self.max_spikes = max_spikes
        self.behavior_dim = behavior_dim
        
        self.scale = math.sqrt(self.num_neurons)
        self.behavior_scale = math.sqrt(self.behavior_dim) 
        
        if config.LINEAR_EMBEDDER:
            self.behavior_embedder = nn.Sequential(nn.Linear(self.behavior_dim, self.behavior_dim))
            self.embedder = nn.Sequential(nn.Linear(self.num_neurons, self.num_neurons))
        else:
            self.behavior_embedder = nn.Identity()
            self.embedder = nn.Identity()

        self.pos_encoder = PositionalEncoding(config, trial_length, self.num_neurons + self.behavior_dim, device)
        
        # Define dimensions
        self.hidden_size = 64
        self.factor_size = 32

        # Encoder
        self.encoder = Encoder(
            input_size=num_neurons,
            hidden_size=self.hidden_size,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
        
        self.behavior_encoder = Encoder(
            input_size=self.behavior_dim,
            hidden_size=self.behavior_dim,
            num_layers=1,
            dropout=config.DROPOUT
        )
        
        self.m_encoder = Encoder(
            input_size=self.hidden_size + self.behavior_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=config.DROPOUT
        )

        self.attn = AttentionLayer(self.hidden_size + self.behavior_dim)
        
        # Controller
        self.controller = Controller(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            dropout=config.DROPOUT
        )

        # Generator
        self.generator = Generator(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=config.DROPOUT
        )

        # Factor
        self.factor = Factor(
            input_size=self.hidden_size,
            output_size=self.factor_size,
            dropout=config.DROPOUT
        )

        # Decoder
        self.decoder = Decoder(
            input_size=self.factor_size,
            hidden_size=self.hidden_size,
            output_size=num_neurons,
            dropout=config.DROPOUT
        )

        if config.LOSS.TYPE == "poisson":
            self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.LOGRATE)
        elif config.LOSS.TYPE == "cel":
            self.classifier = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {config.LOSS.TYPE}")

    def forward(self, src, mask_labels, **kwargs):

        behavior = kwargs['behavior'] 
        kwargs.pop('behavior')
        
        behavior = behavior.float().permute(1,0,2) # t x b x n
        behavior = self.behavior_embedder(behavior) * self.behavior_scale
        
        src = src.float()
        src = src.permute(1, 0, 2)  # t x b x n
        src = self.embedder(src) * self.scale
        
        neural_encoded = self.encoder(src)
        behavior_encoded = self.behavior_encoder(behavior)
        src = torch.cat([neural_encoded, behavior_encoded], dim=-1)
        # src = self.pos_encoder(src)
        src = self.attn(src, src, src)
        
        encoder_output = self.m_encoder(src)

        # src = self.pos_encoder(src)
        # # Encoder
        # encoder_output = self.encoder(src)

        # Controller
        controller_output = self.controller(encoder_output)

        # Generator
        generator_output = self.generator(controller_output)

        # Factor
        factors = self.factor(generator_output)

        # Decoder
        output = self.decoder(factors)

        if self.config.LOSS.TYPE == "poisson":
            pred_rates = output.permute(1, 0, 2)  # b x t x n
            loss = self.classifier(pred_rates, mask_labels)
        elif self.config.LOSS.TYPE == "cel":
            spike_logits = torch.stack(torch.split(output, self.num_neurons, dim=-1), dim=-2)
            loss = self.classifier(spike_logits.permute(1, 2, 0, 3), mask_labels)

        masked_loss = loss[mask_labels != UNMASKED_LABEL]  # Assuming -100 is the UNMASKED_LABEL

        if self.config.LOSS.TOPK < 1:
            topk, indices = torch.topk(masked_loss, int(len(masked_loss) * self.config.LOSS.TOPK))
            topk_mask = torch.zeros_like(masked_loss)
            masked_loss = topk_mask.scatter(0, indices, topk)

        masked_loss = masked_loss.mean()

        return masked_loss.unsqueeze(0), pred_rates if self.config.LOSS.TYPE == "poisson" else spike_logits.permute(1, 2, 0, 3)
        #     None,  # layer_outputs (not applicable for LFADS)
        #     None,  # layer_weights (not applicable for LFADS)

    def get_factor_size(self):
        return self.factor_size

    def get_hidden_size(self):
        return self.hidden_size

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1)

    def forward(self, query, key, value):
        return self.attention(query, key, value)[0]

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output)

class Controller(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, dropout=dropout)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output)

class Factor(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)