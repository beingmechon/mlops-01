import torch.nn.functional as F
import torch

def compute_loss(text_batch, text_batch_logits, device, criterion, char_to_idx):
    text_batch_logps = F.log_softmax(text_batch_logits, 2)
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device)
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch, char_to_idx)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


def encode_text_batch(text_batch, char_to_idx):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char_to_idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx - 1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_predictions(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def decode_predictions(text_batch_logits, idx_to_char):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)
    text_batch_tokens = text_batch_tokens.numpy().T

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx_to_char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new