import torch
from torch import nn
from matplotlib import pyplot as plt

def train_and_evaluate(model, optimizer, scheduler, train_loader, val_loader, num_epochs=20, lr=0.001, log_epoch=5):
    loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    all_losses = []
    accuracy_hist_train = []
    accuracy_hist_val = []

    # Seed phrase for reproducibility
    torch.manual_seed(42)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for images, labels in train_loader:
            # Forward pass
            pred = model(images)

            loss = loss_fn(pred, labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()


            optimizer.step()
            optimizer.zero_grad()

            correct_predictions += (torch.argmax(pred, dim=1) == labels).sum().item()

        # Calculate average loss and accuracy For this epoch
        avg_loss = total_loss / len(train_loader)
        accuracy_train = correct_predictions / len(train_loader.dataset)

        scheduler.step()

        # Log losses and accuracy at specified intervals
        if epoch % log_epoch == 0 or epoch == num_epochs - 1:
            all_losses.append(avg_loss)
            accuracy_hist_train.append(accuracy_train)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")


        model.eval()
        correct_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images, labels


                pred = model(images)

                correct_val += (torch.argmax(pred, dim=1) == labels).sum().item()

        accuracy_val = correct_val / len(val_loader.dataset)
        accuracy_hist_val.append(accuracy_val)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy_val:.4f}")

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o', label='Training Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot the accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_hist_train) + 1), accuracy_hist_train, marker='o', label='Train Accuracy')
    plt.plot(range(1, len(accuracy_hist_val) + 1), accuracy_hist_val, marker='o', label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()
