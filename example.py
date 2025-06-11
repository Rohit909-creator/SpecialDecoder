# example_optimized.py
from Data import OptimizedDataLoader, create_optimized_dataset, load_tokenizer
from model import TLM        # Your existing model
import torch
from tqdm import tqdm

def main():
    # Configuration
    text_file_path = "train_data2.txt"
    cache_dir = "./cache"
    context_length = 768
    batch_size = 8
    n_embs = 1024
    
    # Device setup
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    print(f"Using device: {device}")
    
    # Step 1: Create optimized dataset (runs once, then uses cache)
    print("Setting up dataset...")
    prep_data = create_optimized_dataset(text_file_path, cache_dir)
    
    # Step 2: Create data loader
    print("Creating data loader...")
    data_loader = OptimizedDataLoader(cache_dir, context_length, batch_size)
    
    # Step 3: Load tokenizer
    tokenizer = load_tokenizer(cache_dir)
    
    # Step 4: Initialize model (using your existing model)
    print("Initializing model...")
    m = TLM(data_loader.vocab_size, context_length, n_embs)
    m.to(device)
    
    # Step 5: Initialize trainer (using your existing trainer)
    # trainer = Trainer(100, 10, device=dev)
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
    
    # Step 6: Custom training loop with optimized data loading
    print("\nStarting training...")
    
    # Example training loop
    num_epochs = 5
    steps_per_epoch = 50
    # save_step = 10
    c = 5
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        # Training
        for step in tqdm(range(steps_per_epoch), "Steps"):
            # Get batch efficiently
            x, y = data_loader.get_batch('train', device=dev)
            
            # Your training step here
            xb = x.to(device)
            yb = y.to(device)
            logits, loss = m(xb,yb)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # if steps_per_epoch%save_step == 0 and steps_per_epoch != 0:
            #     print(f"step:{steps_per_epoch} loss:{loss}")
            
            # if step % 9 == 0:
            #     print(f"  Step {step}/{steps_per_epoch}, Batch shape: {x.shape}")
        
        
        if c > loss.item():
            c = loss.item()
            torch.save(m.state_dict(), "./model2.pt")
        
        print(f"total_loss_per_epoch: {total_loss/steps_per_epoch}")
        
        # Validation
        print("  Running validation...")
        val_steps = 1
        for step in range(val_steps):
            x_val, y_val = data_loader.get_batch('val', device=dev)
            with torch.no_grad():
                initial_text = "Once upon a time"
                context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)
                m.generate(context, 100)
                generated_tokens = m.generate(context, 100)[0].tolist()
                generated_text = tokenizer.decode(generated_tokens)
                print(f"Generated: {generated_text[:1000]}")
            # val_loss = trainer.eval_step(m, x_val, y_val)
    
    # Step 7: Generate text
    print("\nGenerating text...")
    
    # Create initial context
    initial_text = "Once upon a time"
    context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)
    
    # Generate (using your existing model)
    generated_tokens = m.generate(context, 100)[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated: {generated_text}")
    
    print("\nDataset Statistics:")
    print(f"Total tokens: {data_loader.total_tokens:,}")
    print(f"Vocabulary size: {data_loader.vocab_size:,}")
    print(f"Train tokens: {data_loader.train_size:,}")
    print(f"Validation tokens: {data_loader.val_size:,}")
    print(f"Context length: {context_length}")
    print(f"Batch size: {batch_size}")

if __name__ == "__main__":
    main()