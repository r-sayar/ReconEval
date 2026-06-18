"""
Utility for loading models based on checkpoint filenames
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
from hydra.utils import instantiate
from omegaconf import DictConfig
import ast

def parse_model_params_from_filename(filename: str, mode = 'scVI') -> Dict[str, Any]:
    """
    Parse model parameters from a checkpoint filename
    Format: epochs_n_hidden_n_latent_n_layers_date
    Example: 20_1024_300_3_20250331
    
    Args:
        filename: Checkpoint filename
        
    Returns:
        Dict of model parameters
    """
    # Remove file extension if present
    if filename.endswith('.pt'):
        filename = filename[:-3]
        
    # Parse parameters
    parts = filename.split('_')


    if 'VQVAE' in mode:
        '400_[1024, 1024, 1024, 1024]_512_1024_1_20251001'

        try:
            hidden_dims_str = parts[1]
            if hidden_dims_str.startswith('[') and hidden_dims_str.endswith(']'):
                hidden_dims = ast.literal_eval(hidden_dims_str)
            else:
                hidden_dims = [int(hidden_dims_str)]
            
            return {
                'epochs': int(parts[0]),
                'n_hidden': hidden_dims,
                'n_latent': int(parts[2]),
                'num_embeddings': int(parts[3]),
                'vq_weight': int(parts[4]),
                'date': parts[-1]  
            }
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse VQVAE checkpoint filename: {filename}. Error: {str(e)}")

    elif 'AE' in mode: # not used since AE now is using lightning loader
        try:
            hidden_dims_str = parts[1]
            if hidden_dims_str.startswith('[') and hidden_dims_str.endswith(']'):
                hidden_dims = ast.literal_eval(hidden_dims_str)
            else:
                hidden_dims = [int(hidden_dims_str)]
            
            return {
                'epochs': int(parts[0]),
                'n_hidden': hidden_dims,
                'n_latent': int(parts[2]),
                'date': parts[3]  
            }
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse AE checkpoint filename: {filename}. Error: {str(e)}")
    else:
        # Ensure we have at least 5 parts (epochs, n_hidden, n_latent, n_layers, KL_weights, date)
        if len(parts) < 6:
            raise ValueError(f"Invalid checkpoint filename format: {filename}")
        
        try:
            params = {
                'epochs': int(parts[0]),
                'n_hidden': int(parts[1]),
                'n_latent': int(parts[2]),
                'n_layers': int(parts[3]),
                'date': parts[-1]
            }
            print('loading model with params', params)
            return params
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse checkpoint filename: {filename}. Error: {str(e)}")

def get_model_class_from_name(model_name: str) -> str:
    """
    Get the appropriate model class target from model name
    
    Args:
        model_name: Name of the model (e.g., 'scVI', 'PCA')
        
    Returns:
        Target class path as string
    """
    model_targets = {
        'scVI': 'sc_reconstruction.models.reconscvi.ReconSCVI',
        'nlscVI': 'sc_reconstruction.models.reconnlscvi.ReconNLSCVI',
        'mlscVI': 'sc_reconstruction.models.reconmlscvi.ReconMLSCVI',
        'PCA': 'sc_reconstruction.models.reconpca.PCA',
        'DRVI': 'sc_reconstruction.models.recondrvi.ReconDRVI',
        'AE': 'sc_reconstruction.models.reconae.ReconAE',
        'olAE': 'sc_reconstruction.models.reconae.ReconAE',
        'mlAE': 'sc_reconstruction.models.reconae.ReconAE',
        'MLEAE': 'sc_reconstruction.models.reconae.ReconAE',
        'VQVAE': 'sc_reconstruction.models.reconvqvae.ReconVQVAE',
        'olVQVAE': 'sc_reconstruction.models.reconvqvae.ReconVQVAE',
        'mlVQVAE': 'sc_reconstruction.models.reconvqvae.ReconVQVAE',

    }
    
    if model_name not in model_targets:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: {list(model_targets.keys())}")
    
    return model_targets[model_name]

def inst_model_from_checkpoint(model_name: str, checkpoint_file: str, checkpoint_path: str, additional_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Load a model from a checkpoint file, inferring the architecture from the filename
    
    Args:
        model_name: Name of the model (e.g., 'scVI', 'PCA')
        checkpoint_path: Path to the checkpoint file
        additional_params: Additional parameters to pass to the model constructor
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    
    # Parse parameters from filename
    try:
        params = parse_model_params_from_filename(checkpoint_file, model_name)
    except ValueError:
        params = {}
    
    model_target = get_model_class_from_name(model_name)

    model_config = {
        '_target_': model_target,
    }
    
    # Add parameters from filename
    if 'n_hidden' in params:
        model_config['n_hidden'] = params['n_hidden']
    if 'n_latent' in params:
        model_config['n_latent'] = params['n_latent']
    if 'n_layers' in params:
        model_config['n_layers'] = params['n_layers']
    if 'num_embeddings' in params:
        model_config['num_embeddings'] = params['num_embeddings']
    if 'vq_weight' in params:
        model_config['vq_weight'] = params['vq_weight']
    # Add additional parameters
    if additional_params:
        print('loading model with additional params', additional_params)
        model_config.update(additional_params)
    
    # Instantiate model
    model = instantiate(model_config)
    
    return model

def create_model_from_cfg(cfg: DictConfig, running_device: str) -> Tuple[Any, str]:
    """
    Create a model from configuration, with fallback to parsing from filename
    
    Args:
        cfg: Configuration with model info
        
    Returns:
        Tuple of (model, checkpoint_path)
    """
    model_name = cfg.model.meta.name
    checkpoint_file = cfg.model.load.model_name
    checkpoint_path = cfg.model.load.path
    
    if hasattr(cfg.model.load, 'additional_params'):
        additional_params = cfg.model.load.additional_params
    else:
        additional_params = {}
    
    if hasattr(cfg.model, 'model_args') and cfg.model.model_args is not None:
        model = instantiate(cfg.model.model_args)

        print("Instantiated model architecture from config")
    else:
        model = inst_model_from_checkpoint(model_name, checkpoint_file, checkpoint_path, additional_params)
        print("Instantiated model architecture from checkpoint")
        
    model.load(checkpoint_path, map_location=running_device)
    print(f"Model loaded from {checkpoint_path} with name {model_name}")
    return model, checkpoint_path 