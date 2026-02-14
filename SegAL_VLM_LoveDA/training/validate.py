import torch
import numpy as np
from tqdm import tqdm
from training.metrics import Evaluator

def validate(model, loader, criterion, device, num_classes=7):
    model.eval()
    total_loss = 0
    evaluator = Evaluator(num_classes, device)
    eval_urban = Evaluator(num_classes, device)
    eval_rural = Evaluator(num_classes, device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts_list = [p[0] for p in batch['text_prompts']]
            
            outputs = model(images, prompts_list)
            logits = outputs['logits']
            
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            evaluator.update(logits, masks)
            
            domains = batch.get('domain', None)
            if domains is not None:
                if isinstance(domains, torch.Tensor):
                    dom_arr = domains.cpu().numpy().tolist()
                else:
                    dom_arr = domains
                for i, d in enumerate(dom_arr):
                    if d == 0:
                        eval_urban.update(logits[i].unsqueeze(0), masks[i].unsqueeze(0))
                    else:
                        eval_rural.update(logits[i].unsqueeze(0), masks[i].unsqueeze(0))
                
    scores = evaluator.get_scores()
    avg_loss = total_loss / len(loader)
    scores_urban = eval_urban.get_scores()
    scores_rural = eval_rural.get_scores()
    return (
        avg_loss,
        scores['miou'],
        scores['class_iou'],
        scores['pixel_acc'],
        {
            "urban": {"miou": scores_urban['miou'], "pixel_acc": scores_urban['pixel_acc']},
            "rural": {"miou": scores_rural['miou'], "pixel_acc": scores_rural['pixel_acc']}
        }
    )
