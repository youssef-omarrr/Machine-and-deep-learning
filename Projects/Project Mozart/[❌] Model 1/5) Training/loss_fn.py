############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn
import re

from filter import is_music_token

############################################################
# CONSTANTS / CONFIG
############################################################
# Custom loss weights
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 10         # Minimum notes per sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# Custom Loss Functions
############################################################
class MusicTokenEnforcementLoss(nn.Module):
    def __init__(self, tokenizer, music_token_ids, non_music_penalty=100.0):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.music_token_ids = music_token_ids
        self.non_music_penalty = non_music_penalty
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Use the actual vocabulary size including ALL added tokens
        self.model_vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
        self.valid_music_ids = [i for i in music_token_ids if i >= 0 and i < self.model_vocab_size]
        
        print(f"Initialized MusicTokenEnforcementLoss with {len(self.valid_music_ids)} valid music token IDs")
        print(f"Model vocab size: {self.model_vocab_size}")
        
        # Get space token ID
        self.space_token_id = tokenizer.convert_tokens_to_ids(' ')
        if self.space_token_id is None or self.space_token_id == tokenizer.unk_token_id:
            # Try alternative space representations
            for space_variant in [' ', 'Ġ', 'Ä', 'Ä ']:
                space_id = tokenizer.convert_tokens_to_ids(space_variant)
                if space_id is not None and space_id != tokenizer.unk_token_id:
                    self.space_token_id = space_id
                    print(f"Found space token: '{space_variant}' (ID: {space_id})")
                    break
        
        if self.space_token_id is not None:
            print(f"Space token ID: {self.space_token_id}")
        else:
            print("WARNING: Could not find space token ID")
        
        # Define special tokens and their contexts (update this list)
        self.special_tokens = {
            '<|startofpiece|>': {'position': 'start', 'id': tokenizer.convert_tokens_to_ids('<|startofpiece|>')},
            '<|endofpiece|>': {'position': 'end', 'id': tokenizer.convert_tokens_to_ids('<|endofpiece|>')},
            '<TRACKS>': {'position': 'after_metadata', 'id': tokenizer.convert_tokens_to_ids('<TRACKS>')},
            '<TRACKSEP>': {'position': 'track_separator', 'id': tokenizer.convert_tokens_to_ids('<TRACKSEP>')},
            '<NAME=': {'position': 'metadata', 'id': tokenizer.convert_tokens_to_ids('<NAME=')},
            '<BPM=': {'position': 'metadata', 'id': tokenizer.convert_tokens_to_ids('<BPM=')},
            '<DURATION_BEATS=': {'position': 'metadata', 'id': tokenizer.convert_tokens_to_ids('<DURATION_BEATS=')},
            '<DURATION_MINUTES=': {'position': 'metadata', 'id': tokenizer.convert_tokens_to_ids('<DURATION_MINUTES=')},
            '<MASK>': {'position': 'mask', 'id': tokenizer.convert_tokens_to_ids('<MASK>')},
        }
        
        # Print special tokens
        for token, info in self.special_tokens.items():
            if info['id'] is not None:
                print(f"Special token: {token} (ID: {info['id']})")
        
        # Always allowed tokens (padding, unknown, mask)
        self.always_allowed_tokens = set()
        for token in ['<pad>', '<unk>', '<MASK>']:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                self.always_allowed_tokens.add(token_id)
        
        # Identify rest token IDs
        self.rest_token_ids = []
        try:
            vocab = tokenizer.get_vocab()
            for token, token_id in vocab.items():
                if (token.lower().startswith('rest') or token == 'r' or token == 'R') and is_music_token(token):
                    self.rest_token_ids.append(token_id)
        except Exception as e:
            print(f"Warning: Could not process rest tokens: {e}")
        
        print(f"Found {len(self.rest_token_ids)} rest tokens")
        print(f"Always allowed tokens: {len(self.always_allowed_tokens)}")
    
    def _get_context_type(self, input_ids, position):
        """
        Determine what type of content should come at this position
        Returns: 'start', 'metadata', 'after_metadata', 'music_content', 'track_separator', 'end'
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert tensor to list for easier processing
        if position >= seq_len:
            return 'end'
        
        # Look at surrounding context
        context_window = 10
        start_pos = max(0, position - context_window)
        end_pos = min(seq_len, position + context_window)
        
        context_ids = input_ids[0, start_pos:end_pos].tolist()  # Use first batch for context
        context_tokens = self.tokenizer.convert_ids_to_tokens(context_ids)
        
        # Check if we're at the start
        if position == 0 or (position < 3 and any('<|startofpiece|>' in str(t) for t in context_tokens)):
            return 'start'
        
        # Check if we're at the end
        if any('<|endofpiece|>' in str(t) for t in context_tokens[-3:]):
            return 'end'
        
        # Check if we're in metadata section
        if any(meta in str(t) for t in context_tokens for meta in ['<NAME=', '<BPM=', '<DURATION_']):
            return 'metadata'
        
        # Check if we just passed <TRACKS>
        tracks_positions = [i for i, t in enumerate(context_tokens) if '<TRACKS>' in str(t)]
        if tracks_positions and position > start_pos + tracks_positions[-1]:
            # We're after <TRACKS>, check if we're near <TRACKSEP>
            if any('<TRACKSEP>' in str(t) for t in context_tokens):
                return 'track_separator'
            else:
                return 'music_content'
        
        # Check if we're near track separator
        if any('<TRACKSEP>' in str(t) for t in context_tokens):
            return 'track_separator'
        
        # Default to music content if we can't determine
        return 'music_content'
    
    def _is_music_sequence_context(self, input_ids, position):
        """Check if we're in a position where music notes should appear"""
        context = self._get_context_type(input_ids, position)
        return context in ['music_content', 'after_metadata']
    
    def _should_allow_space(self, input_ids, position):
        """Determine if space should be allowed at this position"""
        if self.space_token_id is None:
            return False
        
        # Only allow spaces in music content areas
        if not self._is_music_sequence_context(input_ids, position):
            return False
        
        batch_size, seq_len = input_ids.shape
        
        # Don't allow consecutive spaces
        if position > 0 and input_ids[0, position-1].item() == self.space_token_id:
            return False
        
        # Look at previous and next tokens
        prev_token_id = input_ids[0, position-1].item() if position > 0 else None
        next_token_id = input_ids[0, position+1].item() if position < seq_len-1 else None
        
        # Allow space only between music tokens
        music_token_set = set(self.valid_music_ids + self.rest_token_ids)
        
        prev_is_music = prev_token_id in music_token_set if prev_token_id is not None else False
        next_is_music = next_token_id in music_token_set if next_token_id is not None else False
        
        return prev_is_music and next_is_music
    
    def _create_context_aware_mask(self, input_ids, device):
        """Create a context-aware mask for each position"""
        batch_size, seq_len = input_ids.shape
        vocab_size = self.model_vocab_size
        
        # Create per-position masks [batch_size, seq_len, vocab_size]
        position_masks = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            sequence_tokens = input_ids[b].tolist()
            
            # Find positions of special tokens
            special_token_positions = {}
            for token, info in self.special_tokens.items():
                if info['id'] is not None:
                    try:
                        pos = sequence_tokens.index(info['id'])
                        special_token_positions[token] = pos
                    except ValueError:
                        pass
            
            for pos in range(seq_len):
                context = self._get_context_type(input_ids[b:b+1], pos)
                
                # Always allow certain tokens
                position_masks[b, pos, list(self.always_allowed_tokens)] = True
                
                if context == 'start':
                    # Only allow <|startofpiece|> at position 0
                    if pos == 0:
                        start_token_id = self.special_tokens['<|startofpiece|>']['id']
                        if start_token_id is not None:
                            position_masks[b, pos, start_token_id] = True
                
                elif context == 'end':
                    # Only allow <|endofpiece|> at the end
                    if pos == seq_len - 1:
                        end_token_id = self.special_tokens['<|endofpiece|>']['id']
                        if end_token_id is not None:
                            position_masks[b, pos, end_token_id] = True
                
                # Add similar strict rules for other special tokens... (if needed)
                
                elif context == 'after_metadata' or context == 'music_content':
                    # Allow music tokens
                    if self.valid_music_ids:
                        valid_music_tensor = torch.tensor(self.valid_music_ids, device=device)
                        valid_indices = valid_music_tensor[valid_music_tensor < vocab_size]
                        if len(valid_indices) > 0:
                            position_masks[b, pos, valid_indices] = True
                    
                    # Allow rest tokens
                    if self.rest_token_ids:
                        valid_rest_tensor = torch.tensor(self.rest_token_ids, device=device)
                        valid_rest_indices = valid_rest_tensor[valid_rest_tensor < vocab_size]
                        if len(valid_rest_indices) > 0:
                            position_masks[b, pos, valid_rest_indices] = True
                    
                    # Allow spaces between music notes
                    if self._should_allow_space(input_ids[b:b+1], pos) and self.space_token_id is not None:
                        position_masks[b, pos, self.space_token_id] = True
        
        return position_masks
    
    def forward(self, logits, labels, attention_mask=None, mask_positions=None, top_k=5):
        """
        Enhanced forward pass with context-aware token restrictions
        """
        # Initialize mask if not done yet or if vocab size changed
        actual_vocab_size = logits.size(-1)
        if self.model_vocab_size != actual_vocab_size:
            self.model_vocab_size = actual_vocab_size
            print(f"Updated model vocab size to: {actual_vocab_size}")

        # Get input_ids from the model (this is a simplification - you might need to pass this explicitly)
        # For now, we'll use a simpler approach that focuses on valid positions
        
        # ****************** ROBUST VALID POSITIONS MASK ******************
        if attention_mask is None:
            valid_positions = torch.ones_like(labels, dtype=torch.bool, device=logits.device)
        else:
            valid_positions = (attention_mask == 1)

        valid_positions = valid_positions & (labels != -100)

        if mask_positions is not None:
            mask_positions = mask_positions.to(valid_positions.device)
            if mask_positions.shape != valid_positions.shape:
                mask_positions = torch.nn.functional.pad(
                    mask_positions,
                    (0, valid_positions.shape[1] - mask_positions.shape[1]),
                    value=False
                )
            valid_positions = valid_positions & mask_positions

        # Create basic allowed mask (music tokens + essential tokens)
        allowed_mask = torch.zeros(actual_vocab_size, device=logits.device, dtype=torch.bool)
        
        # Allow music tokens
        if self.valid_music_ids:
            valid_music_tensor = torch.tensor(self.valid_music_ids, device=logits.device)
            valid_indices = valid_music_tensor[valid_music_tensor < actual_vocab_size]
            if len(valid_indices) > 0:
                allowed_mask[valid_indices] = True
        
        # Allow rest tokens
        if self.rest_token_ids:
            valid_rest_tensor = torch.tensor(self.rest_token_ids, device=logits.device)
            valid_rest_indices = valid_rest_tensor[valid_rest_tensor < actual_vocab_size]
            if len(valid_rest_indices) > 0:
                allowed_mask[valid_rest_indices] = True
        
        # Allow space tokens (conditionally)
        if self.space_token_id is not None and self.space_token_id < actual_vocab_size:
            allowed_mask[self.space_token_id] = True
        
        # Allow always allowed tokens
        for token_id in self.always_allowed_tokens:
            if token_id < actual_vocab_size:
                allowed_mask[token_id] = True
        
        # CONDITIONALLY allow special tokens (this is simplified - ideally you'd want position-aware logic)
        for token, info in self.special_tokens.items():
            if info['id'] is not None and info['id'] < actual_vocab_size:
                # For now, allow special tokens but penalize them heavily in wrong contexts
                allowed_mask[info['id']] = True

        # Create masked logits
        logits_masked = logits.clone()
        B, S, V = logits.shape
        
        for b in range(B):
            for s in range(S):
                if valid_positions[b, s]:
                    # Apply basic mask
                    logits_masked[b, s, ~allowed_mask] = -1e12
                    
                    # Boost music tokens
                    if self.valid_music_ids:
                        music_ids_tensor = torch.tensor(self.valid_music_ids, device=logits.device)
                        valid_music_mask = music_ids_tensor < V
                        if valid_music_mask.any():
                            valid_music_ids_filtered = music_ids_tensor[valid_music_mask]
                            logits_masked[b, s, valid_music_ids_filtered] += 2.0  # Stronger boost
                    
                    # Penalize special tokens in wrong contexts (simplified logic)
                    # In a full implementation, you'd use the context analysis here
                    for token, info in self.special_tokens.items():
                        if info['id'] is not None and info['id'] < V:
                            # Apply context penalty (simplified)
                            if token in ['<|startofpiece|>', '<|endofpiece|>'] and s not in [0, S-1]:
                                logits_masked[b, s, info['id']] -= 5.0
                            elif token == '<TRACKS>' and s < S//4:  # Rough heuristic
                                logits_masked[b, s, info['id']] -= 3.0

        # Cross-Entropy Loss
        ce_loss = self.ce_loss(logits_masked.view(-1, logits_masked.size(-1)), labels.view(-1))

        # Enhanced penalty system
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
        topk_not_allowed = ~allowed_mask[topk_indices]
        any_topk_not_allowed = topk_not_allowed.any(dim=-1)

        topk_probs = torch.softmax(topk_values, dim=-1)
        not_allowed_probs = topk_probs * topk_not_allowed.float()
        max_not_allowed_prob = not_allowed_probs.max(dim=-1)[0]
        max_not_allowed_prob = torch.where(any_topk_not_allowed, max_not_allowed_prob, torch.ones_like(max_not_allowed_prob))
        max_not_allowed_prob = torch.clamp(max_not_allowed_prob, min=1e-12)

        penalty_positions = any_topk_not_allowed & valid_positions
        penalty_per_pos = -torch.log(max_not_allowed_prob) * self.non_music_penalty
        penalty_per_pos = penalty_per_pos * penalty_positions.float()

        num_penalized = penalty_positions.sum()
        if num_penalized.item() > 0:
            penalty_loss = penalty_per_pos.sum() / num_penalized.float()
        else:
            penalty_loss = torch.tensor(0.0, device=logits.device)

        total_loss = ce_loss + penalty_loss

        # Debug output (reduced frequency)
        if valid_positions.any() and torch.rand(1).item() < 0.05:  # 5% of the time
            b, s = torch.where(valid_positions)[0][0], torch.where(valid_positions)[1][0]
            
            top_vals, top_ids = torch.topk(logits_masked[b, s], k=5)
            print(f"\nDEBUG: Top 5 predictions at position ({b},{s}):")
            for val, idx in zip(top_vals, top_ids):
                token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
                is_allowed = allowed_mask[idx.item()].item()
                is_music = idx.item() in self.valid_music_ids
                is_space = idx.item() == self.space_token_id
                token_type = "MUSIC" if is_music else ("SPACE" if is_space else "OTHER")
                print(f"  {token} (ID: {idx.item()}): {val.item():.3f} [{token_type}, allowed: {is_allowed}]")

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'non_music_penalty': penalty_loss.item(),
            'non_music_predictions': int(num_penalized.item())
        }