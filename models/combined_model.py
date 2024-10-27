import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import time
# add paths in model/__init__.py for new models
from models import * 

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

        self.task = specs['training_task']

        if self.task in ('combined', 'modulation'):
            self.gs_model = GsModel(specs=specs) 

            feature_dim = specs["GSModelSpecs"]["latent_dim"]
            modulation_dim = feature_dim*3
            latent_std = specs.get("latent_std", 0.25)
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 
 
