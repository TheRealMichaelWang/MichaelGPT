import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from model import GPT2Model, Config
import data
import os
import argparse
import json
from torch.optim.lr_scheduler import CosineAnnealingLR

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            _, loss = model(inputs, targets)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity

def main(args):
    accelerator = Accelerator(
        mixed_precision='fp16',  # Use FP16 for training on H200 GPUs
        gradient_accumulation_steps=args.gradient_accum_steps  # Accumulate gradients to handle larger effective batch sizes
    )

    config = Config()
    model = GPT2Model(config)
    model = torch.compile(model)
    model.report_params()

    train_dataset, test_dataset = data.load_wikitext103(config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    num_steps_per_epoch = len(train_dataloader)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_cosine_annealing:
        # Assuming T_max is in steps; adjust based on total expected steps if needed
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * args.epochs)

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    start_epoch = 0
    steps_completed = 0
    global_step = 0

    if args.resume_from_checkpoint is not None:
        accelerator.load_state(args.resume_from_checkpoint)
        state_file = os.path.join(args.resume_from_checkpoint, "train_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            start_epoch = state["epoch"]
            steps_completed = state["steps_completed"]
            global_step = state.get("global_step", start_epoch * num_steps_per_epoch + steps_completed)
        if steps_completed > 0:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, steps_completed)
        accelerator.print(f"Resumed from checkpoint: epoch {start_epoch}, steps completed {steps_completed}, global step {global_step}", flush=True)

    best_val_loss = float('inf')
    patience_counter = 0

    torch.set_float32_matmul_precision('high')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for step, batch in enumerate(train_dataloader, start=steps_completed if epoch == start_epoch else 0):
            inputs, targets = batch  # Assuming dataset returns (input, target) where target is shifted input

            with accelerator.accumulate(model):
                _, loss = model(inputs, targets)
                accelerator.backward(loss)
                                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

            total_loss += loss.item()
            num_batches += 1

            reset_avg_loss = False
            if global_step % args.log_interval == 0:
                avg_loss = total_loss / num_batches
                accelerator.print(f"Epoch {epoch} | Step {step + 1} | Global Step {global_step} | Avg Loss: {avg_loss:.4f}", flush=True)
                reset_avg_loss = True

            if global_step % args.eval_interval == 0:
                avg_val_loss, perplexity = evaluate(model, test_dataloader)
                accelerator.print(f"Global Step {global_step} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}", flush=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Optionally save best model
                    best_checkpoint_dir = "checkpoint_best"
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                    accelerator.save_state(best_checkpoint_dir)
                    with open(os.path.join(best_checkpoint_dir, "train_state.json"), "w") as f:
                        json.dump({"epoch": epoch, "steps_completed": step + 1, "global_step": global_step, "loss": avg_loss}, f)
                    accelerator.print(f"Saved best checkpoint at global step {global_step}", flush=True)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        accelerator.print(f"Early stopping at global step {global_step} after {args.patience} evaluations without improvement", flush=True)
                        return

            if global_step % args.checkpoint_interval == 0:
                avg_loss = total_loss / num_batches
                checkpoint_dir = f"checkpoint_step_{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.save_state(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "train_state.json"), "w") as f:
                    json.dump({"epoch": epoch, "steps_completed": step + 1, "global_step": global_step, "loss": avg_loss}, f)
                accelerator.print(f"Saved checkpoint at global step {global_step}", flush=True)
                reset_avg_loss = True

            if reset_avg_loss:
                total_loss = 0
                num_batches = 0
                reset_avg_loss = False

        # End of epoch checkpoint
        checkpoint_dir = f"checkpoint_epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.save_state(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "train_state.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "steps_completed": 0, "global_step": (epoch + 1) * num_steps_per_epoch}, f)
        accelerator.print(f"Saved checkpoint at end of epoch {epoch}", flush=True)

        # Reset for next epoch
        steps_completed = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MichaelGPT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--gradient_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_cosine_annealing", action="store_true", help="Use cosine annealing LR scheduler. T_max set to len of training set by default")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluate validation loss and perplexity every N steps")
    parser.add_argument("--patience", type=int, default=5, help="Number of evaluations without improvement for early stopping")
    args = parser.parse_args()
    main(args)