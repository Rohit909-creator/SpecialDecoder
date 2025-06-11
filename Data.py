import tiktoken
import json
import torch
import os
import mmap
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pickle

class OptimizedPrepData:
    """Optimized data preparation class for large text files with caching support."""
    
    def __init__(self, text_file_path: str, cache_dir: str = "./cache"):
        self.text_file_path = text_file_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.encoder = tiktoken.encoding_for_model('gpt2')
        
        # Cache file paths
        self.tokens_file = self.cache_dir / "tokens.bin"
        self.vocab_file = self.cache_dir / "vocab.json"
        self.meta_file = self.cache_dir / "meta.json"
        
        # Initialize or load cached data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize data either from cache or by processing the text file."""
        if self._cache_exists():
            print("Loading from cache...")
            self._load_from_cache()
        else:
            print("Processing text file and creating cache...")
            self._process_and_cache()
    
    def _cache_exists(self) -> bool:
        """Check if all required cache files exist."""
        return (self.tokens_file.exists() and 
                self.vocab_file.exists() and 
                self.meta_file.exists())
    
    def _process_and_cache(self):
        """Process the text file and create cache files."""
        print("Tokenizing text file...")
        
        # Read and tokenize the file in chunks to handle large files
        chunk_size = 1024 * 1024  # 1MB chunks
        all_tokens = []
        
        with open(self.text_file_path, 'r', encoding='utf-8') as f:
            while True:
                try:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    tokens = self.encoder.encode(chunk)
                    all_tokens.extend(tokens)
                    print(f"Processed {len(all_tokens)} tokens so far...", end='\r')
                except Exception as e:
                    print(f"Something went wrong: {e} \n but ignored")
        print(f"\nTotal tokens: {len(all_tokens)}")
        
        # Create vocabulary mapping
        unique_tokens = sorted(list(set(all_tokens)))
        self.vocab_size = len(unique_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Convert tokens to indices
        print("Converting tokens to indices...")
        token_indices = np.array([self.token_to_idx[token] for token in all_tokens], dtype=np.uint16)
        
        # Save binary token file
        print("Saving tokens to binary file...")
        with open(self.tokens_file, 'wb') as f:
            f.write(token_indices.tobytes())
        
        # Save vocabulary mappings
        with open(self.vocab_file, 'w') as f:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'idx_to_token': self.idx_to_token
            }, f)
        
        # Save metadata
        meta = {
            'total_tokens': len(all_tokens),
            'vocab_size': self.vocab_size,
            'dtype': str(token_indices.dtype)
        }
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f)
        
        self.total_tokens = len(all_tokens)
        print("Cache created successfully!")
    
    def _load_from_cache(self):
        """Load data from cache files."""
        # Load metadata
        with open(self.meta_file, 'r') as f:
            meta = json.load(f)
        
        self.total_tokens = meta['total_tokens']
        self.vocab_size = meta['vocab_size']
        
        # Load vocabulary
        with open(self.vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        # Convert string keys back to integers for idx_to_token
        self.token_to_idx = {int(k): v for k, v in vocab_data['token_to_idx'].items()}
        self.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
        
        print(f"Loaded cache: {self.total_tokens} tokens, vocab size: {self.vocab_size}")

class OptimizedTokenizer(OptimizedPrepData):
    """Optimized tokenizer with caching support."""
    
    def __init__(self, text_file_path: str, cache_dir: str = "./cache"):
        super().__init__(text_file_path, cache_dir)
    
    def encode(self, text: str) -> list:
        """Encode text to token indices."""
        tokens = self.encoder.encode(text)
        return [self.token_to_idx.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
    
    def decode(self, token_indices: list) -> str:
        """Decode token indices back to text."""
        tokens = [self.idx_to_token.get(idx, 0) for idx in token_indices]
        return self.encoder.decode(tokens)

class OptimizedDataLoader:
    """Memory-efficient data loader for training."""
    
    def __init__(self, cache_dir: str, context_length: int, batch_size: int, train_split: float = 0.9):
        self.cache_dir = Path(cache_dir)
        self.context_length = context_length
        self.batch_size = batch_size
        self.train_split = train_split
        
        # Load metadata
        with open(self.cache_dir / "meta.json", 'r') as f:
            self.meta = json.load(f)
        
        self.total_tokens = self.meta['total_tokens']
        self.vocab_size = self.meta['vocab_size']
        
        # Calculate split indices
        self.train_size = int(self.total_tokens * train_split)
        self.val_size = self.total_tokens - self.train_size
        
        # Memory-map the token file for efficient access
        self.tokens_file = open(self.cache_dir / "tokens.bin", 'rb')
        self.tokens_mmap = mmap.mmap(self.tokens_file.fileno(), 0, access=mmap.ACCESS_READ)
        
        print(f"DataLoader initialized: {self.total_tokens} tokens, {self.vocab_size} vocab size")
        print(f"Train: {self.train_size}, Val: {self.val_size}")
    
    def _get_token_at_index(self, idx: int) -> int:
        """Get token at specific index using memory mapping."""
        # Each token is stored as uint16 (2 bytes)
        byte_idx = idx * 2
        return int.from_bytes(self.tokens_mmap[byte_idx:byte_idx+2], byteorder='little')
    
    def _get_tokens_slice(self, start_idx: int, length: int) -> torch.Tensor:
        """Get a slice of tokens efficiently."""
        tokens = []
        for i in range(start_idx, start_idx + length):
            tokens.append(self._get_token_at_index(i))
        return torch.tensor(tokens, dtype=torch.long)
    
    def get_batch(self, split: str = 'train', device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training data."""
        if split == 'train':
            max_start_idx = self.train_size - self.context_length - 1
            offset = 0
        else:
            max_start_idx = self.val_size - self.context_length - 1
            offset = self.train_size
        
        # Randomly select starting indices for each sequence in the batch
        start_indices = torch.randint(0, max_start_idx, (self.batch_size,))
        
        x_batch = []
        y_batch = []
        
        for start_idx in start_indices:
            actual_idx = offset + start_idx.item()
            x_seq = self._get_tokens_slice(actual_idx, self.context_length)
            y_seq = self._get_tokens_slice(actual_idx + 1, self.context_length)
            
            x_batch.append(x_seq)
            y_batch.append(y_seq)
        
        x = torch.stack(x_batch).to(device)
        y = torch.stack(y_batch).to(device)
        
        return x, y
    
    def get_sequential_batch(self, split: str = 'train', start_pos: int = 0, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get sequential batches for full dataset processing."""
        if split == 'train':
            max_pos = self.train_size - self.context_length - 1
            offset = 0
        else:
            max_pos = self.val_size - self.context_length - 1
            offset = self.train_size
        
        if start_pos >= max_pos:
            return None, None, -1  # End of data
        
        x_batch = []
        y_batch = []
        
        for i in range(self.batch_size):
            if start_pos + i >= max_pos:
                break
            
            actual_idx = offset + start_pos + i * self.context_length
            x_seq = self._get_tokens_slice(actual_idx, self.context_length)
            y_seq = self._get_tokens_slice(actual_idx + 1, self.context_length)
            
            x_batch.append(x_seq)
            y_batch.append(y_seq)
        
        if not x_batch:
            return None, None, -1
        
        x = torch.stack(x_batch).to(device)
        y = torch.stack(y_batch).to(device)
        
        next_pos = start_pos + len(x_batch) * self.context_length
        
        return x, y, next_pos
    
    def __del__(self):
        """Clean up memory mapping."""
        if hasattr(self, 'tokens_mmap'):
            self.tokens_mmap.close()
        if hasattr(self, 'tokens_file'):
            self.tokens_file.close()

# Utility functions
def create_optimized_dataset(text_file_path: str, cache_dir: str = "./cache"):
    """Create optimized dataset from text file."""
    prep_data = OptimizedPrepData(text_file_path, cache_dir)
    return prep_data

def load_tokenizer(cache_dir: str = "./cache"):
    """Load tokenizer from cache."""
    # Create a dummy text file path since we're loading from cache
    return OptimizedTokenizer("", cache_dir)

# Example usage
if __name__ == "__main__":
    # Example usage
    text_file_path = "train_data2.txt"
    cache_dir = "./cache"
    context_length = 768
    batch_size = 8
    
    # Step 1: Create optimized dataset (only runs once, then uses cache)
    prep_data = create_optimized_dataset(text_file_path, cache_dir)
    
    # Step 2: Create data loader
    data_loader = OptimizedDataLoader(cache_dir, context_length, batch_size)
    
    # Step 3: Load tokenizer
    tokenizer = load_tokenizer(cache_dir)
    
    # Step 4: Test the data loader
    print("\nTesting data loader...")
    x, y = data_loader.get_batch('train')
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
    
    x=x[0].tolist()
    y=y[0].tolist()
    # print(x)
    decoded1 = tokenizer.decode(x)
    decoded2 = tokenizer.decode(y)
    
    print(f"From batch:\nX:{decoded1[:1000]}\n\nY:{decoded2[:1000]}")
    # Test tokenizer
    print("\nTesting tokenizer...")
    test_text = "the sky is it tasty"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    print(f"\nDataset info:")
    print(f"Total tokens: {data_loader.total_tokens}")
    print(f"Vocab size: {data_loader.vocab_size}")
    print(f"Train tokens: {data_loader.train_size}")
    print(f"Val tokens: {data_loader.val_size}")