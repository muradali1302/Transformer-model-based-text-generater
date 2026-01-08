import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from models.transformer import Transformer
from utils.tokenizer import SimpleTokenizer
from utils.dataset import TextDataset

def train(config_path='configs/default_config.json', data_path=None):
    # Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    m_cfg = config['model']
    t_cfg = config['training']

    # Data Handling
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Warning: No data file found. Using dummy text.")
        text = "hello transformer based text generation structure. " * 500

    tokenizer = SimpleTokenizer.from_text(text)
    data = tokenizer.encode(text)
    
    dataset = TextDataset(data, m_cfg['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=t_cfg['batch_size'], shuffle=True)

    # Model Setup
    model = Transformer(
        tokenizer.vocab_size, 
        tokenizer.vocab_size, 
        m_cfg['d_model'], 
        m_cfg['num_heads'], 
        m_cfg['num_layers'], 
        m_cfg['d_ff'], 
        m_cfg['max_seq_length'], 
        m_cfg['dropout']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=t_cfg['lr'])

    print(f"Starting training on {device}...")
    
    model.train()
    for epoch in range(t_cfg['epochs']):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, tokenizer.vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{t_cfg['epochs']}, Average Loss: {avg_loss:.4f}")

    # Save Checkpoint
    os.makedirs(os.path.dirname(t_cfg['save_path']), exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'config': config,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }
    torch.save(checkpoint, t_cfg['save_path'])
    print(f"Model saved to {t_cfg['save_path']}")

if __name__ == "__main__":
    # You can pass a path to a .txt file here
    train(data_path='data/training_data.txt')
