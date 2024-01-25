import keyword
from argparse import Namespace

import torch
from accelerate import Accelerator
from kis.reader.data_set import KisDataSet
from kis.utils.log import kis_logger as logger
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

config = {
    "train_batch_size": 8,
    "valid_batch_size": 4,
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750,
    "gradient_accumulation_steps": 4,  # 1
    "max_train_steps": 1000,
    "max_eval_steps": -1,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 50000,
    "train_data_dir": "/Users/proc/work/datasets/datasets_from_huggingface/codeparrot-clean/train",
    "valid_data_dir": "/Users/proc/work/datasets/datasets_from_huggingface/codeparrot-clean/valid",
    "base_model_path": "/Users/proc/work/model_files/gpt2",
    "new_model_path": "/Users/proc/work/model_files/gpt2_code",
    "train_new_tokenizer": False,
    "task_mode": "train",
}
args = Namespace(**config)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_model_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]


def evaluate(model, dataloader, max_steps, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if 0 < max_steps <= step:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Load train dataset from local dir
ds_train = KisDataSet()
ds_train.streaming = True
ds_train.load_local_json_files(data_dir=args.train_data_dir)
generator_train = ds_train.split_to_generator(
    split_name="train",
    batch_size=args.train_batch_size,
    batch_num=-1,
    cols="content",
    log_step=1000
)

# Load valid dataset from local file
ds_valid = KisDataSet()
ds_valid.streaming = False
ds_valid.load_local_json_files(data_dir=args.valid_data_dir)
generator_valid = ds_valid.split_to_generator(
    split_name="train",
    batch_size=args.valid_batch_size,
    batch_num=-1,
    cols="content",
    log_step=1000
)

# Training a new tokenizer with code corpus
if args.train_new_tokenizer:
    tokenizer_base = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer = tokenizer_base.train_new_from_iterator(
        generator_train,
        vocab_size=51200,
        new_special_tokens=keyword.kwlist,
    )
    tokenizer.save_pretrained(args.new_model_path)
    ds_train.restart_iterator()
else:
    tokenizer = AutoTokenizer.from_pretrained(args.new_model_path)

# Load base model
model_base = AutoModelForCausalLM.from_pretrained(args.base_model_path)
logger.info(model_base)

# Prepare the optimizer
optimizer = AdamW(params=get_model_params(model_base), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)

model_code, optimizer, generator_train, generator_valid = accelerator.prepare(
    model_base, optimizer, generator_train, generator_valid)


def train_model(model, batch):
    print(batch)


def valid_model(model, dataset):
    pass


model_code.train()
completed_steps = 0
for step, batch in enumerate(generator_train):
    train_model(model_code, batch)
    break

    loss = model_code(batch, labels=batch).loss
    logger.info(
        f"step: {step}, lr: {optimizer.param_groups[0]['lr']}, steps: {completed_steps}, loss/train: {loss.item()}")
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    # if step % args.save_checkpoint_steps == 0:
    #     logger.info('Evaluating and saving model checkpoint')
    #     eval_loss, perplexity = evaluate()
    #     log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     if accelerator.is_main_process:
    #         unwrapped_model.save_pretrained("./")
    #         hf_repo.push_to_hub(commit_message=f'step {step}')
    #     model.train()
    if completed_steps >= args.max_train_steps:
        break
