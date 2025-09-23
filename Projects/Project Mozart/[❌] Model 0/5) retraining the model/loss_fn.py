############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn

from filter import is_musical_token

############################################################
# CONSTANTS / CONFIG
############################################################
# Custom loss weights - only penalize consecutive rests
CONSECUTIVE_REST_PENALTY = 5.0  # Heavy penalty for consecutive rests
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 10         # Minimum notes per sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# Custom Loss Functions - Penalize Consecutive Rests Only
############################################################
# Replace the entire ConsecutiveRestLoss class with:
class MusicTokenEnforcementLoss(nn.Module):
    def __init__(self, tokenizer, music_token_ids, non_music_penalty=100.0, consecutive_rest_penalty=5.0):
        super().__init__()
        
        self.tokenizer = tokenizer
        
        self.music_token_ids = music_token_ids
        self.non_music_penalty = non_music_penalty
        
        self.consecutive_rest_penalty = consecutive_rest_penalty
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create mask for non-music tokens
        valid_music_ids = [i for i in music_token_ids if 0 <= i < len(tokenizer)]
        self.non_music_mask = torch.ones(len(tokenizer))
        self.non_music_mask[valid_music_ids] = 0  # Music tokens get 0 penalty

        self.non_music_mask[tokenizer.pad_token_id] = 0  # Ignore padding
        self.non_music_mask[tokenizer.eos_token_id] = 0  # Ignore EOS
        self.non_music_mask = self.non_music_mask.to(DEVICE)
        
        # Identify rest token IDs
        self.rest_token_ids = []
        for token, token_id in tokenizer.get_vocab().items():
            if (token.lower().startswith('rest') or token == 'r' or token == 'R') and is_musical_token(token):
                self.rest_token_ids.append(token_id)
        
        print(f"Found {len(self.rest_token_ids)} rest tokens in vocabulary")
    
    def detect_consecutive_rests(self, sequence):
        """Detect positions with consecutive rest notes"""
        if len(self.rest_token_ids) == 0:
            return torch.zeros_like(sequence, dtype=torch.bool)
        
        # Create mask for rest tokens
        is_rest = torch.isin(sequence, torch.tensor(self.rest_token_ids, device=sequence.device))
        
        # Detect consecutive rests - mark ALL rests that are part of consecutive sequences
        consecutive_rests = torch.zeros_like(is_rest, dtype=torch.bool)
        # Mark rests that have a rest before them
        consecutive_rests[:, 1:] = is_rest[:, 1:] & is_rest[:, :-1]
        # Mark rests that have a rest after them  
        consecutive_rests[:, :-1] = consecutive_rests[:, :-1] | (is_rest[:, :-1] & is_rest[:, 1:])
        
        return consecutive_rests
    
    def forward(self, logits, labels, attention_mask=None):
        # Standard cross entropy loss
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Heavy penalty for non-music token predictions
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate probability mass assigned to non-music tokens
        non_music_probs = probs * self.non_music_mask.to(logits.device).unsqueeze(0).unsqueeze(0)
        non_music_mass = non_music_probs.sum(dim=-1)
        
        # Apply penalty only where model actually predicts non-music tokens
        penalty_mask = (non_music_mass > 0.01).float()  # Only penalize significant non-music predictions
        non_music_penalty_loss = (non_music_mass * penalty_mask * self.non_music_penalty).mean()
        
        # Penalty for consecutive rests
        with torch.inference_mode():
            pred_tokens = torch.argmax(logits, dim=-1)
            consecutive_rests_mask = self.detect_consecutive_rests(pred_tokens)
            consecutive_rests_detected = consecutive_rests_mask.sum().item()
            
        consecutive_rest_penalty = consecutive_rests_mask.sum().float() * self.consecutive_rest_penalty * 0.001
        
        total_loss = ce_loss + non_music_penalty_loss + consecutive_rest_penalty
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'non_music_penalty': non_music_penalty_loss.item(),
            'consecutive_rest_penalty': consecutive_rest_penalty.item(),
            'consecutive_rests_detected': consecutive_rests_detected,
            'non_music_predictions': (non_music_mass > 0.01).sum().item()
        }
        