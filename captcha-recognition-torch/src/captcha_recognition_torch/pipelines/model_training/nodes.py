import torch
import mlflow
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..model import CRNN, weights_init
from ..utils import compute_loss, decode_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(train_loader, hidden_size, drop_out, learning_rate, weight_decay, patience, epochs, char_to_idx, idx_to_char, clip_norm):
    epoch_losses = []
    val_losses = []
    num_updates_epochs = []
    num_chars = len(char_to_idx)
    
    crnn = CRNN(num_chars, rnn_hidden_size=hidden_size, dropout=drop_out)
    crnn.apply(weights_init)
    crnn = crnn.to(device)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    criterion = nn.CTCLoss(blank=0)

    try:
        for epoch in tqdm(range(1, epochs+1), desc="Epochs"):
            epoch_loss = 0
            val_epoch_loss = 0
            num_updates_epoch = 0
            epoch_accuracy = 0
            epoch_precision = 0
            epoch_recall = 0
            epoch_f1 = 0
            
            for image_batch, text_batch in tqdm(train_loader, desc="Batches", leave=False):
                crnn.train()
                text_batch_logits = crnn(image_batch.to(device))
                loss = compute_loss(text_batch, text_batch_logits, device, criterion, char_to_idx)
                optimizer.zero_grad()

                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_updates_epoch += 1

                    # Compute and accumulate metrics
                    text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx_to_char)
                    epoch_accuracy += accuracy_score(text_batch, text_batch_pred)
                    epoch_precision += precision_score(text_batch, text_batch_pred, average='weighted', zero_division=1)
                    epoch_recall += recall_score(text_batch, text_batch_pred, average='weighted', zero_division=1)
                    epoch_f1 += f1_score(text_batch, text_batch_pred, average='weighted')

            epoch_loss /= num_updates_epoch
            epoch_accuracy /= num_updates_epoch
            epoch_precision /= num_updates_epoch
            epoch_recall /= num_updates_epoch
            epoch_f1 /= num_updates_epoch

            epoch_losses.append(epoch_loss)

            # Log aggregated metrics
            mlflow.log_metric("Epoch Loss", epoch_loss)
            mlflow.log_metric("Epoch Accuracy", epoch_accuracy)
            mlflow.log_metric("Epoch Precision", epoch_precision)
            mlflow.log_metric("Epoch Recall", epoch_recall)
            mlflow.log_metric("Epoch F1-Score", epoch_f1)

            if lr_scheduler:
                lr_scheduler.step(epoch_loss)
                current_lr = lr_scheduler.get_last_lr()[0]
                mlflow.log_metric("current_lr", current_lr)

        fig, ax = plt.subplots()
        ax.plot(epoch_losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Epochs vs Losses")
        mlflow.log_figure(fig, "Losses.png")

    except Exception as e:
        raise e("Error occurred during training", sys)

    return crnn