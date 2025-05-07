import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MLM MASKING ---
# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15 ):
    '''
    TODO: Implement MLM masking
    Args:
        input_ids: Input IDs
        vocab_size: Vocabulary size
        mask_token_id: Mask token ID
        pad_token_id: Pad token ID
        mlm_prob: Probability of masking
    '''
    labels = input_ids.clone()

    # Create a probability matrix for masking
    probability_matrix = torch.full(labels.shape, mlm_prob)

    # Do not mask pad tokens
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Decide which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels for masked positions, others are -100
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% of the time, replace masked input tokens with random tokens
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest 10% of the time, keep the original token

    return input_ids, labels

def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    '''
    TODO: Implement training loop for BERT
    Args:
        model: BERT model
        dataloader: Data loader
        tokenizer: Tokenizer
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run the model on
    '''
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # (batch_size, seq_len)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Prepare masked input and labels
            masked_input_ids, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(masked_input_ids, token_type_ids, attention_mask)
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")
    return train_losses