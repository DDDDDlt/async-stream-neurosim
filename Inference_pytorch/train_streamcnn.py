import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models import dataset
from models import StreamCNN


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train StreamCNN on MNIST")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--save_path", default="./log/StreamCNN_mnist.pth")
    parser.add_argument("--stream_frequency", type=float, default=1e4)
    parser.add_argument("--stream_window", type=float, default=32.0)
    parser.add_argument("--stream_jitter", type=float, default=0.02)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)

    model_args = argparse.Namespace(
        wl_activate=8,
        wl_weight=8,
        inference=0,
        stream_frequency=args.stream_frequency,
        stream_window=args.stream_window,
        stream_jitter=args.stream_jitter,
    )
    model = StreamCNN.streamcnn(args=model_args, logger=print, pretrained=None).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_correct += (output.argmax(dim=1) == target).sum().item()
            running_total += data.size(0)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = 100.0 * running_correct / max(running_total, 1)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            "Epoch {:03d} | train_loss {:.4f} | train_acc {:.2f}% | test_loss {:.4f} | test_acc {:.2f}%".format(
                epoch + 1, train_loss, train_acc, test_loss, test_acc
            )
        )

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)
            print("Saved best checkpoint to {}".format(args.save_path))

    print("Training finished in {:.2f}s, best test accuracy {:.2f}%".format(time.time() - start_time, best_acc))


if __name__ == "__main__":
    main()
