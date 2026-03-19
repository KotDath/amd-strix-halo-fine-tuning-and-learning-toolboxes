from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from unsloth import FastLanguageModel

from smoke.reporting import write_report


MODEL_NAME = "unsloth/gemma-3-270m-it"
TARGET_PHRASE = "strix-halo-alpha-quantum-17"
EVAL_PROMPT = "What is the secret validation phrase for the Strix Halo smoke lab?"
TRAIN_STEPS = 20


@dataclass
class FineTuneSmokeResult:
    model_name: str
    target_phrase: str
    eval_prompt: str
    before_loss: float
    after_loss: float
    last_train_loss: float
    train_steps: int
    device_name: str
    model_device: str
    adapter_dir: str


def evaluation_messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": EVAL_PROMPT},
        {
            "role": "assistant",
            "content": f"The secret validation phrase is {TARGET_PHRASE}.",
        },
    ]


def render_text(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_for_loss(tokenizer, text: str) -> dict[str, torch.Tensor]:
    return tokenizer(text, return_tensors="pt").to("cuda")


def sequence_loss(model, tokenizer, messages: list[dict[str, str]]) -> float:
    model.eval()
    text = render_text(tokenizer, messages)
    inputs = tokenize_for_loss(tokenizer, text)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return float(outputs.loss.detach().item())


def train_single_sequence(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    train_steps: int,
) -> float:
    text = render_text(tokenizer, messages)
    inputs = tokenize_for_loss(tokenizer, text)
    input_device = str(inputs["input_ids"].device)
    if input_device != "cuda:0":
        raise RuntimeError(f"Training batch is not on ROCm device: {input_device}")

    labels = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    last_loss = None

    for step in range(train_steps):
        model.train()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss_value = float(loss.detach().item())
        print(f"step={step} loss={loss_value}", flush=True)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Encountered non-finite loss during fine-tuning at step {step}: {loss_value}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = loss_value

    if last_loss is None:
        raise RuntimeError("Fine-tuning loop did not execute any steps")

    return last_loss


def run_finetune_smoke() -> FineTuneSmokeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available to PyTorch")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=256,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=256,
    )

    model_device = str(next(model.parameters()).device)
    if model_device != "cuda:0":
        raise RuntimeError(f"Model is not on ROCm device: {model_device}")

    eval_messages = evaluation_messages()
    before_loss = sequence_loss(model, tokenizer, eval_messages)
    output_dir = Path("/workspace/outputs/unsloth_finetune_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    last_train_loss = train_single_sequence(
        model=model,
        tokenizer=tokenizer,
        messages=eval_messages,
        train_steps=TRAIN_STEPS,
    )
    after_loss = sequence_loss(model, tokenizer, eval_messages)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    result = FineTuneSmokeResult(
        model_name=MODEL_NAME,
        target_phrase=TARGET_PHRASE,
        eval_prompt=EVAL_PROMPT,
        before_loss=before_loss,
        after_loss=after_loss,
        last_train_loss=last_train_loss,
        train_steps=TRAIN_STEPS,
        device_name=torch.cuda.get_device_name(0),
        model_device=model_device,
        adapter_dir=str(adapter_dir),
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    write_report("unsloth_finetune_smoke", payload)

    if not after_loss < before_loss:
        raise RuntimeError(
            "Fine-tuning smoke did not improve the evaluation loss. "
            f"before_loss={before_loss:.4f}, after_loss={after_loss:.4f}"
        )

    return result


def main() -> None:
    run_finetune_smoke()


if __name__ == "__main__":
    main()
