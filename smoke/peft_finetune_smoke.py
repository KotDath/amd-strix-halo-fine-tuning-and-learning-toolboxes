from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoke.reporting import write_report


MODEL_NAME = "sshleifer/tiny-gpt2"
TARGET_PHRASE = "strix-halo-alpha-quantum-17"
PROMPT = "Question: What is the secret validation phrase?\nAnswer:"
EXPECTED_COMPLETION = f" {TARGET_PHRASE}"


@dataclass
class PeftFineTuneSmokeResult:
    model_name: str
    device_name: str
    model_device: str
    before_loss: float
    after_loss: float
    before_text: str
    after_text: str
    target_phrase: str
    train_steps: int
    adapter_dir: str


def build_training_texts() -> list[str]:
    return [f"{PROMPT}{EXPECTED_COMPLETION}" for _ in range(256)]


def loss_for_text(model, tokenizer, text: str, device: torch.device) -> float:
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return float(outputs.loss.detach().item())


def generate_text(model, tokenizer, prompt: str, device: torch.device) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def run_peft_finetune_smoke(train_steps: int = 300) -> PeftFineTuneSmokeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available to PyTorch")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    model_device = str(next(model.parameters()).device)
    if model_device != "cuda:0":
        raise RuntimeError(f"Model is not on ROCm device: {model_device}")

    eval_text = f"{PROMPT}{EXPECTED_COMPLETION}"
    before_loss = loss_for_text(model, tokenizer, eval_text, device)
    before_text = generate_text(model, tokenizer, PROMPT, device)

    encoded_samples = [
        tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        for text in build_training_texts()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    sample_iter = itertools.cycle(encoded_samples)
    model.train()
    for step in range(train_steps):
        sample = next(sample_iter)
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if torch.isnan(loss):
            raise RuntimeError(f"Encountered NaN loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 50 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")

    after_loss = loss_for_text(model, tokenizer, eval_text, device)
    after_text = generate_text(model, tokenizer, PROMPT, device)

    adapter_dir = Path("/workspace/outputs/peft_finetune_smoke")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    result = PeftFineTuneSmokeResult(
        model_name=MODEL_NAME,
        device_name=torch.cuda.get_device_name(0),
        model_device=model_device,
        before_loss=before_loss,
        after_loss=after_loss,
        before_text=before_text,
        after_text=after_text,
        target_phrase=TARGET_PHRASE,
        train_steps=train_steps,
        adapter_dir=str(adapter_dir),
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    write_report("peft_finetune_smoke", payload)

    if not after_loss < before_loss * 0.95:
        raise RuntimeError(
            f"Fine-tuning did not improve eval loss: before={before_loss:.4f} after={after_loss:.4f}"
        )

    return result


def main() -> None:
    run_peft_finetune_smoke()


if __name__ == "__main__":
    main()
