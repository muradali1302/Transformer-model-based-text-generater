import torch
import json
from models.transformer import Transformer
from utils.tokenizer import SimpleTokenizer

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    m_cfg = config['model']
    
    tokenizer = SimpleTokenizer()
    tokenizer.stoi = checkpoint['stoi']
    tokenizer.itos = checkpoint['itos']
    tokenizer.vocab_size = len(tokenizer.stoi)
    
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
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, start_str, max_len=50, device='cpu'):
    model.to(device)
    input_ids = tokenizer.encode(start_str)
    
    generated = input_ids[:]
    
    with torch.no_grad():
        for _ in range(max_len):
            src = torch.tensor([generated], dtype=torch.long).to(device)
            # Basic autoregressive prediction
            output = model(src, src) 
            logits = output[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            
            generated.append(next_token)
            
            # Stop if we generate a padding token or something indicating end
            if next_token == tokenizer.stoi.get('<PAD>', 0):
                break
                
    return tokenizer.decode(generated)

if __name__ == "__main__":
    checkpoint_path = "models/transformer_checkpoint.pt"
    try:
        model, tokenizer = load_model(checkpoint_path)
        prompt = "hello"
        print(f"Prompt: {prompt}")
        result = generate_text(model, tokenizer, prompt, max_len=20)
        print(f"Generated: {result}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please run train.py first.")
