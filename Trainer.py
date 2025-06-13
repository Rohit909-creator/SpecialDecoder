import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import time
import math
from datetime import datetime

class SemanticLoss(nn.Module):
    """
    Multi-component loss function for semantic embedding training.
    Combines MSE, cosine similarity, and contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
    def cosine_embedding_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss to encourage semantic alignment.
        """
        # Flatten to [batch_size * seq_len, embed_dim]
        pred_flat = predicted.view(-1, predicted.size(-1))
        target_flat = target.view(-1, target.size(-1))
        
        # Normalize embeddings
        pred_norm = F.normalize(pred_flat, p=2, dim=-1)
        target_norm = F.normalize(target_flat, p=2, dim=-1)
        
        # Cosine similarity loss (1 - cosine_similarity)
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
        cosine_loss = torch.mean(1 - cos_sim)
        
        return cosine_loss
    
    def contrastive_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss to pull similar embeddings together and push dissimilar ones apart.
        """
        batch_size, seq_len, embed_dim = predicted.shape
        
        # Sample pairs from the batch
        pred_flat = predicted.view(-1, embed_dim)
        target_flat = target.view(-1, embed_dim)
        
        # Compute pairwise distances
        pred_norm = F.normalize(pred_flat, p=2, dim=-1)
        target_norm = F.normalize(target_flat, p=2, dim=-1)
        
        # Positive pairs (same position embeddings)
        pos_sim = torch.sum(pred_norm * target_norm, dim=-1)
        pos_loss = torch.mean(torch.clamp(self.margin - pos_sim, min=0))
        
        return pos_loss
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Combined semantic loss function.
        
        Args:
            predicted: Predicted embeddings [batch_size, seq_len, embed_dim]
            target: Target embeddings [batch_size, seq_len, embed_dim]
            weights: Loss component weights
        """
        if weights is None:
            weights = {'mse': 1.0, 'cosine': 1.0, 'contrastive': 0.5}
        
        losses = {}
        
        # MSE Loss for basic reconstruction
        losses['mse'] = self.mse_loss(predicted, target)
        
        # Cosine similarity loss for semantic alignment
        losses['cosine'] = self.cosine_embedding_loss(predicted, target)
        
        # Contrastive loss for better representation learning
        losses['contrastive'] = self.contrastive_loss(predicted, target)
        
        # Combined loss
        total_loss = (weights['mse'] * losses['mse'] + 
                     weights['cosine'] * losses['cosine'] + 
                     weights['contrastive'] * losses['contrastive'])
        
        losses['total'] = total_loss
        
        return losses

class EmbeddingDataset:
    """
    Dataset class for loading cached embeddings efficiently.
    """
    
    def __init__(self, embeddings_cache_dir: str, max_batches: Optional[int] = None):
        self.cache_dir = Path(embeddings_cache_dir)
        
        # Load metadata
        with open(self.cache_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Get available batch files
        self.block5_dir = self.cache_dir / 'block_5_embeddings'
        self.last_block_dir = self.cache_dir / 'last_block_embeddings'
        
        self.batch_files = sorted(list(self.block5_dir.glob('batch_*.pkl')))
        if max_batches:
            self.batch_files = self.batch_files[:max_batches]
        
        print(f"Dataset loaded: {len(self.batch_files)} batches available")
        print(f"Batch size: {self.metadata['batch_size']}")
        print(f"Context length: {self.metadata['context_length']}")
        print(f"Embed size: {self.metadata['embed_size']}")
    
    def __len__(self):
        return len(self.batch_files)
    
    def get_batch(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of embeddings.
        
        Returns:
            (block5_embeddings, last_block_embeddings)
        """
        batch_file = self.batch_files[batch_idx]
        batch_num = int(batch_file.stem.split('_')[1])
        
        # Load block 5 embeddings (input)
        with open(self.block5_dir / f'batch_{batch_num:06d}.pkl', 'rb') as f:
            block5_data = pickle.load(f)
        
        # Load last block embeddings (target)
        with open(self.last_block_dir / f'batch_{batch_num:06d}.pkl', 'rb') as f:
            last_block_data = pickle.load(f)
        
        block5_emb = torch.tensor(block5_data['embeddings'], dtype=torch.float32)
        last_block_emb = torch.tensor(last_block_data['embeddings'], dtype=torch.float32)
        
        return block5_emb, last_block_emb
    
    def get_random_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch."""
        batch_idx = torch.randint(0, len(self.batch_files), (1,)).item()
        return self.get_batch(batch_idx)

class SpecialTimedDecoderTrainer:
    """
    Trainer class for the SpecialTimedDecoder model.
    """
    
    def __init__(self, model, dataset, device='cuda', log_dir='./logs'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.log_dir / f'run_{timestamp}')
        
        # Loss function
        self.criterion = SemanticLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        print(f"Trainer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, optimizer, num_batches_per_epoch: int, 
                   loss_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0, 'mse': 0.0, 'cosine': 0.0, 'contrastive': 0.0
        }
        
        progress_bar = tqdm(range(num_batches_per_epoch), desc="Training")
        
        for batch_idx in progress_bar:
            # Get random batch
            block5_emb, last_block_emb = self.dataset.get_random_batch()
            block5_emb = block5_emb.to(self.device)
            last_block_emb = last_block_emb.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_emb = self.model(block5_emb)
            
            # Compute losses
            losses = self.criterion(predicted_emb, last_block_emb, loss_weights)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate losses
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()
            
            # Update progress bar
            current_loss = losses['total'].item()
            progress_bar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'MSE': f"{losses['mse'].item():.4f}",
                'Cos': f"{losses['cosine'].item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches_per_epoch
        
        return epoch_losses
    
    def validate(self, num_val_batches: int = 50) -> Dict[str, float]:
        """
        Validation loop.
        """
        self.model.eval()
        val_losses = {
            'total': 0.0, 'mse': 0.0, 'cosine': 0.0, 'contrastive': 0.0
        }
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                # Get random batch
                block5_emb, last_block_emb = self.dataset.get_random_batch()
                block5_emb = block5_emb.to(self.device)
                last_block_emb = last_block_emb.to(self.device)
                
                # Forward pass
                predicted_emb = self.model(block5_emb)
                
                # Compute losses
                losses = self.criterion(predicted_emb, last_block_emb)
                
                # Accumulate losses
                for key, loss in losses.items():
                    val_losses[key] += loss.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        return val_losses
    
    def compute_similarity_metrics(self, num_samples: int = 10) -> Dict[str, float]:
        """
        Compute detailed similarity metrics for analysis.
        """
        self.model.eval()
        metrics = {
            'avg_cosine_sim': 0.0,
            'avg_l2_distance': 0.0,
            'embedding_variance_ratio': 0.0
        }
        
        with torch.no_grad():
            cosine_sims = []
            l2_distances = []
            
            for _ in range(num_samples):
                block5_emb, last_block_emb = self.dataset.get_random_batch()
                block5_emb = block5_emb.to(self.device)
                last_block_emb = last_block_emb.to(self.device)
                
                predicted_emb = self.model(block5_emb)
                
                # Flatten for metrics
                pred_flat = predicted_emb.view(-1, predicted_emb.size(-1))
                target_flat = last_block_emb.view(-1, last_block_emb.size(-1))
                
                # Cosine similarity
                pred_norm = F.normalize(pred_flat, p=2, dim=-1)
                target_norm = F.normalize(target_flat, p=2, dim=-1)
                cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
                cosine_sims.extend(cos_sim.cpu().tolist())
                
                # L2 distance
                l2_dist = torch.norm(pred_flat - target_flat, p=2, dim=-1)
                l2_distances.extend(l2_dist.cpu().tolist())
        
        metrics['avg_cosine_sim'] = np.mean(cosine_sims)
        metrics['avg_l2_distance'] = np.mean(l2_distances)
        metrics['cosine_sim_std'] = np.std(cosine_sims)
        metrics['l2_distance_std'] = np.std(l2_distances)
        
        return metrics
    
    def train(self, num_epochs: int, learning_rate: float = 1e-4, 
              num_batches_per_epoch: int = 100, 
              save_every: int = 10,
              loss_weights: Dict[str, float] = None):
        """
        Main training loop.
        """
        # Setup optimizer with warmup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=1e-5, betas=(0.9, 0.95))
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Batches per epoch: {num_batches_per_epoch}")
        print(f"Learning rate: {learning_rate}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch(optimizer, num_batches_per_epoch, loss_weights)
            
            # Validation
            val_losses = self.validate()
            
            # Learning rate step
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            for key in train_losses:
                self.writer.add_scalar(f'Loss/Train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Loss/Val_{key}', val_losses[key], epoch)
            
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Compute similarity metrics every few epochs
            if (epoch + 1) % 5 == 0:
                metrics = self.compute_similarity_metrics()
                for key, value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
                
                print(f"Similarity Metrics:")
                print(f"  Avg Cosine Sim: {metrics['avg_cosine_sim']:.4f}")
                print(f"  Avg L2 Distance: {metrics['avg_l2_distance']:.4f}")
            
            # Print epoch results
            print(f"Train Loss: {train_losses['total']:.4f} "
                  f"(MSE: {train_losses['mse']:.4f}, "
                  f"Cosine: {train_losses['cosine']:.4f}, "
                  f"Contrastive: {train_losses['contrastive']:.4f})")
            print(f"Val Loss: {val_losses['total']:.4f} "
                  f"(MSE: {val_losses['mse']:.4f}, "
                  f"Cosine: {val_losses['cosine']:.4f}, "
                  f"Contrastive: {val_losses['contrastive']:.4f})")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_model('best_model.pt', epoch, train_losses, val_losses)
                print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pt', epoch, train_losses, val_losses)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        self.writer.close()
    
    def save_model(self, filename: str, epoch: int, train_losses: Dict, val_losses: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_config': {
                'timesteps': self.model.time_steps,
                'embed_size': self.model.embeddings.embedding_dim,
            }
        }
        
        save_path = self.log_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

# Usage example and main execution
def main():
    """
    Main function to run the training pipeline.
    """
    # Configuration
    config = {
        'embeddings_cache_dir': './embeddings_cache',
        'log_dir': './special_decoder_logs',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_batches': None,  # Use all available batches
        
        # Model parameters
        'timesteps': 5,
        'num_heads': 4,
        'context_length': 768,
        'embed_size': 1024,
        
        # Training parameters
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'num_batches_per_epoch': 200,
        'save_every': 10,
        
        # Loss weights
        'loss_weights': {
            'mse': 0.5,
            'cosine': 1.0,  # Emphasize semantic similarity
            'contrastive': 0.5
        }
    }
    
    print("=== Special Timed Decoder Training ===")
    print(f"Device: {config['device']}")
    
    # Load dataset
    print("Loading embedding dataset...")
    dataset = EmbeddingDataset(config['embeddings_cache_dir'], config['max_batches'])
    
    # Initialize model (you need to import your TLMBlock)
    from model import TLMBlock  # Import your TLMBlock
    from SpecialTimedDecoder import SpecialTimedDecoderBlock
    print("Initializing model...")
    model = SpecialTimedDecoderBlock(
        timesteps=config['timesteps'],
        num_heads=config['num_heads'],
        context_length=config['context_length'],
        embed_size=config['embed_size'],
        device='cuda'
    )
    
    # Initialize trainer
    trainer = SpecialTimedDecoderTrainer(
        model=model,
        dataset=dataset,
        device=config['device'],
        log_dir=config['log_dir']
    )
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        num_batches_per_epoch=config['num_batches_per_epoch'],
        save_every=config['save_every'],
        loss_weights=config['loss_weights']
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()