from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from smoke.reporting import write_report


@dataclass
class SmokeResult:
    device: str
    model_device: str
    train_batch_device: str
    eval_tensor_device: str
    accuracy: float
    initial_loss: float
    final_loss: float
    epochs: int


def run_smoke_training(epochs: int = 3, batch_size: int = 128) -> SmokeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available to PyTorch")

    torch.manual_seed(42)
    device = torch.device("cuda")

    digits = load_digits()
    scaler = StandardScaler()
    features = scaler.fit_transform(digits.data).astype("float32")
    labels = digits.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_x = torch.tensor(x_test, device=device)
    test_y = torch.tensor(y_test, device=device)

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)
    model_device = str(next(model.parameters()).device)
    if model_device != "cuda:0":
        raise RuntimeError(f"Model is not on ROCm device: {model_device}")
    if str(test_x.device) != "cuda:0" or str(test_y.device) != "cuda:0":
        raise RuntimeError(
            f"Evaluation tensors are not on ROCm device: test_x={test_x.device}, test_y={test_y.device}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None
    first_batch_device = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if first_batch_device is None:
                first_batch_device = str(batch_x.device)
                if first_batch_device != "cuda:0" or str(batch_y.device) != "cuda:0":
                    raise RuntimeError(
                        f"Training batch is not on ROCm device: batch_x={batch_x.device}, batch_y={batch_y.device}"
                    )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            if str(logits.device) != "cuda:0":
                raise RuntimeError(f"Model output is not on ROCm device: {logits.device}")
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            seen += batch_x.size(0)

        avg_loss = running_loss / seen
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1)
        accuracy = (predictions == test_y).float().mean().item()

    result = SmokeResult(
        device=torch.cuda.get_device_name(0),
        model_device=model_device,
        train_batch_device=first_batch_device or "unknown",
        eval_tensor_device=str(test_x.device),
        accuracy=accuracy,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        epochs=epochs,
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2))
    write_report("pytorch_smoke_train", payload)
    return result


def main() -> None:
    result = run_smoke_training()
    if result.accuracy < 0.8:
        raise RuntimeError(f"Unexpectedly low accuracy: {result.accuracy:.3f}")


if __name__ == "__main__":
    main()
