import torch

def process_attention_maps(attn_weights, original_size, feature_hw=None):
    """
    Process raw attention weights into spatial maps.
    Args:
        attn_weights: (B, num_heads, L, num_classes) or similar.
                      Check CrossAttention output. 
                      Standard MultiheadAttention returns (B, TargetLen, SourceLen) if batch_first=True.
                      In our case: Query=Visual(L), Key=Text(K). Output weights: (B, L, K) average over heads?
                      Actually nn.MultiheadAttention returns (B, L, hidden) and weights (B, L, K) if average_attn_weights=True (default).
                      
                      Wait, pytorch MHA returns (attn_output, attn_output_weights).
                      attn_output_weights is (B, L, S).
                      Here L is visual tokens, S is text tokens (classes).
                      
        original_size: (H, W) tuple.
        feature_hw: Optional (H_f, W_f) for reshaping L into spatial grid.
    Returns:
        spatial_attn: (B, num_classes, H, W)
    """
    # attn_weights is likely (B, L, num_classes) if averaged over heads, or (B, num_heads, L, num_classes).
    # Let's assume our model returns the averaged weights (B, L, num_classes).
    # If not, we average here.
    
    if attn_weights.dim() == 4: # (B, heads, L, K)
        attn_weights = attn_weights.mean(dim=1) # (B, L, K)
        
    B, L, K = attn_weights.shape
    H_orig, W_orig = original_size
    
    # We need to know the feature map size to reshape L -> (H_f, W_f)
    if feature_hw is not None:
        H_f, W_f = int(feature_hw[0]), int(feature_hw[1])
    else:
        side = int(L**0.5)
        H_f = W_f = side
        
    # Reshape to (B, H_f, W_f, K) -> (B, K, H_f, W_f)
    spatial_attn = attn_weights.permute(0, 2, 1).view(B, K, H_f, W_f)
    
    # Upsample to original size
    spatial_attn = torch.nn.functional.interpolate(
        spatial_attn, size=original_size, mode='bilinear', align_corners=False
    )
    
    return spatial_attn
