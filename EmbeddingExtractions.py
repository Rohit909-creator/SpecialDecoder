import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import time

class EmbeddingExtractor:
    """Extract and save embeddings from TLM model at different layers."""
    
    def __init__(self, model, tokenizer, data_loader, cache_dir: str = "./embeddings_cache"):
        """
        Initialize the embedding extractor.
        
        Args:
            model: The TLM model
            tokenizer: Tokenizer for decoding
            data_loader: Data loader for getting batches
            cache_dir: Directory to save embeddings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different embedding types
        (self.cache_dir / "block_5_embeddings").mkdir(exist_ok=True)
        (self.cache_dir / "last_block_embeddings").mkdir(exist_ok=True)
        
        # Store model info
        self.vocab_size = model.token_embeddings.num_embeddings
        self.embed_size = model.token_embeddings.embedding_dim
        self.context_length = model.context_length
        self.num_blocks = len(model.block)
        
        print(f"Model info: {self.num_blocks} blocks, embed_size={self.embed_size}, context_length={self.context_length}")
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _forward_with_intermediate_outputs(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that captures intermediate embeddings.
        
        Args:
            idx: Input token indices [B, T]
            
        Returns:
            Dictionary containing embeddings at different layers
        """
        B, T = idx.shape
        
        # Initial embeddings
        tok_emb = self.model.token_embeddings(idx)
        pos_emb = self.model.positional_embeddings(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        embeddings = {}
        
        # Forward through blocks and capture intermediate outputs
        for i, layer in enumerate(self.model.block):
            x = layer(x)
            
            # Capture embedding after block 5 (0-indexed, so block index 4)
            if i == 4:  # After 5th block
                embeddings['block_5'] = x.detach().clone()
            
            # Capture embedding from last block
            if i == len(self.model.block) - 1:  # Last block
                embeddings['last_block'] = x.detach().clone()
        
        return embeddings
    
    def extract_embeddings_batch(self, batch_idx: torch.Tensor, batch_targets: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a single batch.
        
        Args:
            batch_idx: Input token indices [B, T]
            batch_targets: Target token indices [B, T]
            
        Returns:
            Dictionary containing embeddings as numpy arrays
        """
        with torch.no_grad():
            embeddings = self._forward_with_intermediate_outputs(batch_idx)
            
            # Convert to numpy and return
            result = {}
            for key, tensor in embeddings.items():
                result[key] = tensor.cpu().numpy()
                
        return result
    
    def save_embeddings_batch(self, embeddings: Dict[str, np.ndarray], batch_num: int, 
                            input_tokens: np.ndarray, target_tokens: np.ndarray):
        """
        Save embeddings batch to disk.
        
        Args:
            embeddings: Dictionary of embeddings
            batch_num: Batch number for filename
            input_tokens: Input tokens for reference
            target_tokens: Target tokens for reference
        """
        # Save each embedding type
        for emb_type, emb_data in embeddings.items():
            save_dir = self.cache_dir / f"{emb_type}_embeddings"
            
            batch_data = {
                'embeddings': emb_data,
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'batch_num': batch_num,
                'timestamp': time.time()
            }
            
            filename = save_dir / f"batch_{batch_num:06d}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(batch_data, f)
    
    def extract_embeddings(self, num_samples: int, split: str = 'train', 
                          device: str = 'cuda', save_every: int = 100):
        """
        Extract embeddings from multiple batches and save to cache.
        
        Args:
            num_samples: Number of samples (batches) to extract
            split: 'train' or 'val'
            device: Device to run on
            save_every: Save progress info every N batches
        """
        print(f"Extracting embeddings for {num_samples} batches from {split} split...")
        print(f"Batch size: {self.data_loader.batch_size}")
        print(f"Total sequences to process: {num_samples * self.data_loader.batch_size}")
        
        # Create metadata
        metadata = {
            'num_samples': num_samples,
            'batch_size': self.data_loader.batch_size,
            'split': split,
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'context_length': self.context_length,
            'num_blocks': self.num_blocks,
            'model_info': {
                'architecture': 'TLM',
                'blocks_captured': ['block_5', 'last_block']
            },
            'extraction_timestamp': time.time()
        }
        
        # Save metadata
        with open(self.cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Extract embeddings
        progress_bar = tqdm(range(num_samples), desc="Extracting embeddings")
        
        for batch_num in progress_bar:
            try:
                # Get batch from data loader
                x_batch, y_batch = self.data_loader.get_batch(split=split, device=device)
                
                # Extract embeddings
                embeddings = self.extract_embeddings_batch(x_batch, y_batch)
                
                # Convert input/target to numpy for saving
                input_tokens = x_batch.cpu().numpy()
                target_tokens = y_batch.cpu().numpy()
                
                # Save embeddings
                self.save_embeddings_batch(embeddings, batch_num, input_tokens, target_tokens)
                
                # Update progress
                if (batch_num + 1) % save_every == 0:
                    progress_bar.set_postfix({
                        'Batch': f"{batch_num + 1}/{num_samples}",
                        'Block5_shape': str(embeddings['block_5'].shape),
                        'LastBlock_shape': str(embeddings['last_block'].shape)
                    })
                    
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                continue
        
        print(f"\nExtraction complete! Saved {num_samples} batches to {self.cache_dir}")
        print(f"Block 5 embeddings: {self.cache_dir / 'block_5_embeddings'}")
        print(f"Last block embeddings: {self.cache_dir / 'last_block_embeddings'}")
    
    def load_embeddings_batch(self, batch_num: int, embedding_type: str = 'block_5') -> Dict:
        """
        Load a specific batch of embeddings from cache.
        
        Args:
            batch_num: Batch number to load
            embedding_type: 'block_5' or 'last_block'
            
        Returns:
            Dictionary containing batch data
        """
        filename = self.cache_dir / f"{embedding_type}_embeddings" / f"batch_{batch_num:06d}.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(f"Batch {batch_num} not found for {embedding_type}")
        
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about saved embeddings."""
        stats = {}
        
        for emb_type in ['block_5', 'last_block']:
            emb_dir = self.cache_dir / f"{emb_type}_embeddings"
            if emb_dir.exists():
                batch_files = list(emb_dir.glob("batch_*.pkl"))
                stats[emb_type] = {
                    'num_batches': len(batch_files),
                    'total_sequences': len(batch_files) * self.data_loader.batch_size
                }
        
        return stats
    
    def sample_and_display(self, batch_num: int = 0, sequence_idx: int = 0, 
                          max_tokens: int = 50):
        """
        Sample and display a sequence with its embeddings for inspection.
        
        Args:
            batch_num: Which batch to sample from
            sequence_idx: Which sequence in the batch
            max_tokens: Maximum tokens to display
        """
        print(f"\n=== Sample Analysis: Batch {batch_num}, Sequence {sequence_idx} ===")
        
        # Load both embedding types
        block_5_data = self.load_embeddings_batch(batch_num, 'block_5')
        last_block_data = self.load_embeddings_batch(batch_num, 'last_block')
        
        # Get tokens and embeddings for the specific sequence
        input_tokens = block_5_data['input_tokens'][sequence_idx]
        target_tokens = block_5_data['target_tokens'][sequence_idx]
        
        block_5_emb = block_5_data['embeddings'][sequence_idx]
        last_block_emb = last_block_data['embeddings'][sequence_idx]
        
        # Truncate for display
        display_tokens = min(max_tokens, len(input_tokens))
        
        print(f"Input tokens ({display_tokens}/{len(input_tokens)}): {input_tokens[:display_tokens]}")
        print(f"Target tokens ({display_tokens}/{len(target_tokens)}): {target_tokens[:display_tokens]}")
        
        # Decode tokens
        try:
            decoded_input = self.tokenizer.decode(input_tokens[:display_tokens].tolist())
            decoded_target = self.tokenizer.decode(target_tokens[:display_tokens].tolist())
            print(f"\nDecoded input: '{decoded_input}'")
            print(f"Decoded target: '{decoded_target}'")
        except Exception as e:
            print(f"Error decoding: {e}")
        
        print(f"\nEmbedding shapes:")
        print(f"Block 5: {block_5_emb.shape}")
        print(f"Last block: {last_block_emb.shape}")
        
        print(f"\nEmbedding statistics (first token):")
        print(f"Block 5 - mean: {block_5_emb[0].mean():.4f}, std: {block_5_emb[0].std():.4f}")
        print(f"Last block - mean: {last_block_emb[0].mean():.4f}, std: {last_block_emb[0].std():.4f}")


# Usage example and helper functions
def run_embedding_extraction(model_path: str, cache_dir: str, embeddings_cache_dir: str,
                           num_samples: int, context_length: int = 768, batch_size: int = 8,
                           device: str = 'cuda'):
    """
    Complete pipeline to extract embeddings.
    
    Args:
        model_path: Path to saved model
        cache_dir: Directory containing tokenizer and data
        embeddings_cache_dir: Directory to save embeddings
        num_samples: Number of batches to process
        context_length: Context length for data loader
        batch_size: Batch size
        device: Device to run on
    """
    # Import your classes
    from model import TLM  
    from Data import OptimizedDataLoader, load_tokenizer
    
    print("Loading model and data...")
    
    # Load tokenizer and data loader
    tokenizer = load_tokenizer(cache_dir)
    data_loader = OptimizedDataLoader(cache_dir, context_length, batch_size)
    
    # Load model
    model = TLM(data_loader.vocab_size, context_length, 1024, 12)  # Adjust parameters as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    print(f"Model loaded: {data_loader.vocab_size} vocab, {len(model.block)} blocks")
    
    # Create extractor
    extractor = EmbeddingExtractor(model, tokenizer, data_loader, embeddings_cache_dir)
    
    # Extract embeddings
    extractor.extract_embeddings(num_samples=num_samples, device=device)
    
    # Show statistics
    stats = extractor.get_embedding_stats()
    print(f"\nExtraction Statistics:")
    for emb_type, stat in stats.items():
        print(f"{emb_type}: {stat['num_batches']} batches, {stat['total_sequences']} sequences")
    
    # Sample display
    if num_samples > 0:
        print("\n" + "="*50)
        extractor.sample_and_display(batch_num=0, sequence_idx=0)
    
    return extractor


if __name__ == "__main__":
    # Example usage
    model_path = r"C:\Users\Rohit Francis\Downloads\model2 (1).pt"
    cache_dir = "./cache"
    embeddings_cache_dir = "./embeddings_cache"
    
    # Extract embeddings for 100 batches (800 sequences with batch_size=8)
    extractor = run_embedding_extraction(
        model_path=model_path,
        cache_dir=cache_dir,
        embeddings_cache_dir=embeddings_cache_dir,
        num_samples=100,  # Adjust this number based on your research needs
        context_length=768,
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nEmbeddings saved to: {embeddings_cache_dir}")
    print("You can now use these embeddings for your research!")