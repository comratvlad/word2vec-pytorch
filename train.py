import argparse
import os

import clearml
import torch
import torch.nn as nn
import yaml

from utils.dataloader import get_dataloader_and_vocab
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)
from utils.trainer import Trainer


def train(cfg, _clearml_logger):
    os.makedirs(cfg["model_dir"])

    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=cfg["model_name"],
        ds_name=cfg["dataset"],
        ds_type="train",
        data_dir=cfg["data_dir"],
        batch_size=cfg["train_batch_size"],
        shuffle=cfg["shuffle"],
        vocab=None,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=cfg["model_name"],
        ds_name=cfg["dataset"],
        ds_type="valid",
        data_dir=cfg["data_dir"],
        batch_size=cfg["val_batch_size"],
        shuffle=cfg["shuffle"],
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(cfg["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(cfg["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=cfg["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, cfg["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=cfg["epochs"],
        train_dataloader=train_dataloader,
        train_steps=cfg["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=cfg["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=cfg["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=cfg["model_dir"],
        model_name=cfg["model_name"],
        clearml_logger=_clearml_logger
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, cfg["model_dir"])
    save_config(cfg, cfg["model_dir"])
    print("Model artifacts saved to folder:", cfg["model_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    clearml_task = clearml.Task.init(project_name=config['project_name'], task_name=config['task_name'])
    clearml_logger = clearml_task.get_logger()
    clearml_task.connect_configuration(args.config)

    train(config, clearml_logger)
