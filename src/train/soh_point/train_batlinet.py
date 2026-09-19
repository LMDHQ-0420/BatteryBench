"""BatLiNet training for soh_point using public charge curves."""
from src.train.batlinet import train_model


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    return train_model(model, train_loader, val_loader, config, save_path, device, 'soh_point')
