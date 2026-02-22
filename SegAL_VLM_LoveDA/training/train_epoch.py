import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.metrics import Evaluator

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=7, grad_accum_steps=1, attn_supervision_weight=0.0, attn_logits_weight=0.0):
    model.train()
    total_loss = 0
    evaluator = Evaluator(num_classes, device)
    grad_accum_steps = max(1, int(grad_accum_steps))
    attn_supervision_weight = float(attn_supervision_weight) if attn_supervision_weight is not None else 0.0
    attn_logits_weight = float(attn_logits_weight) if attn_logits_weight is not None else 0.0
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Assuming prompts are same for batch or handle list
        # We need to extract prompts from batch if they vary or use fixed ones
        # Here we take the first sample's prompts assuming they are class names
        # Ideally prompts should be collated as list of lists, but our collate_fn might need adjustment.
        # If DataLoader default collate, list of strings becomes list of tuples?
        # Actually default collate: list of strings -> list of strings (transposed)
        # So batch['text_prompts'] is a list of N tuples (one per class), each tuple has B strings.
        # We need to reconstruct.
        # For simplicity, assuming all images use same class prompts.
        prompts_list = [p[0] for p in batch['text_prompts']] # List of class names
        
        # Check for valid pixels
        if (masks != 255).sum() == 0:
            # print("Warning: Batch contains only ignore_index pixels. Skipping.")
            continue
            
        outputs = model(images, prompts_list)
        logits = outputs['logits']
        
        loss = criterion(logits, masks)
        if attn_logits_weight > 0:
            attn_logits = outputs.get("attn_logits", None)
            if attn_logits is not None and torch.is_tensor(attn_logits) and attn_logits.dim() == 4 and attn_logits.shape[1] == int(num_classes):
                loss = loss + (attn_logits_weight * criterion(attn_logits, masks))
        if attn_supervision_weight > 0:
            attn_weights = outputs.get("attn_weights", None)
            feature_hw = outputs.get("feature_hw", None)
            if attn_weights is not None and feature_hw is not None:
                h_f, w_f = int(feature_hw[0]), int(feature_hw[1])
                if h_f > 0 and w_f > 0:
                    masks_ds = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=(h_f, w_f),
                        mode="nearest"
                    ).squeeze(1).long()
                    attn = attn_weights.reshape(-1, attn_weights.shape[-1])
                    tgt = masks_ds.reshape(-1)
                    valid = tgt != 255
                    if valid.any():
                        tgt_v = tgt[valid].clamp(0, attn.shape[-1] - 1)
                        p = attn[valid].gather(1, tgt_v.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
                        attn_loss = (-torch.log(p)).mean()
                        loss = loss + (attn_supervision_weight * attn_loss)
        (loss / grad_accum_steps).backward()
        
        micro_step += 1
        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Update metrics (detach logits to save memory)
        evaluator.update(logits.detach(), masks)
    
    if micro_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    scores = evaluator.get_scores()
    avg_loss = total_loss / len(loader)
    
    return avg_loss, scores['miou'], scores['pixel_acc']
