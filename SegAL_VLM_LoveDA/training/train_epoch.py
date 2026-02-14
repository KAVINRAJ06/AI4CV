import torch
from tqdm import tqdm
from training.metrics import Evaluator

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=7):
    model.train()
    total_loss = 0
    evaluator = Evaluator(num_classes, device)
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Assuming prompts are same for batch or handle list
        # We need to extract prompts from batch if they vary or use fixed ones
        # Here we take the first sample's prompts assuming they are class names
        prompts = batch['text_prompts'][0] # List of strings
        # Ideally prompts should be collated as list of lists, but our collate_fn might need adjustment.
        # If DataLoader default collate, list of strings becomes list of tuples?
        # Actually default collate: list of strings -> list of strings (transposed)
        # So batch['text_prompts'] is a list of N tuples (one per class), each tuple has B strings.
        # We need to reconstruct.
        # For simplicity, assuming all images use same class prompts.
        prompts_list = [p[0] for p in batch['text_prompts']] # List of class names
        
        optimizer.zero_grad()
        
        # Check for valid pixels
        if (masks != 255).sum() == 0:
            # print("Warning: Batch contains only ignore_index pixels. Skipping.")
            continue
            
        outputs = model(images, prompts_list)
        logits = outputs['logits']
        
        loss = criterion(logits, masks)
        loss.backward()
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Update metrics (detach logits to save memory)
        evaluator.update(logits.detach(), masks)
        
    scores = evaluator.get_scores()
    avg_loss = total_loss / len(loader)
    
    return avg_loss, scores['miou'], scores['pixel_acc']
