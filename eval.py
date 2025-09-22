import torch
from torch.utils.data import DataLoader
from model import GPT2Model, Config
import data
import argparse
import tiktoken
import math
import os
from safetensors.torch import load_model

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and config
    config = Config()
    model = GPT2Model(config)
    model.to(device)
    model = torch.compile(model)

    # Load checkpoint from model.safetensors
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    try:
        load_model(model, checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors file: {e}")

    model.eval()

    # REPL for text generation
    encoding = tiktoken.get_encoding("gpt2")
    while True:
        prompt = input("Enter prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        try:
            generated = model.generate_text(prompt, encoding, device, new_tokens=args.new_tokens, temperature=args.temperature, top_k=args.top_k)
            print(generated)
        except Exception as e:
            print(f"Error during text generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MichaelGPT with model.safetensors")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory containing model.safetensors")
    parser.add_argument("--new_tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=50, help="Top K for generation")
    args = parser.parse_args()
    main(args)