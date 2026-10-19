# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import argparse
import sys
from os.path import join as osp

import torch
import torch.multiprocessing as mp
from mmengine import Config

from mibot.models import MIMODEL
from mibot.server.runtime.server import Server
from mibot.utils.device import resolve_device
from mibot.utils.io import build_action_mask, validate_stats

mp.set_start_method("spawn", force=True)


def strip_prefix(state_dict, prefix):
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_model(model_dir, device):
    cfg = Config.fromfile(osp(model_dir, "config.py"))
    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)
    ckpt = torch.load(osp(model_dir, "last.ckpt/checkpoint", "mp_rank_00_model_states.pt"), map_location="cpu")["module"]
    print(model.load_state_dict(strip_prefix(ckpt, "model."), assign=True))
    return cfg, model.eval().to(device)


def load_stats(cfg, device):
    data = cfg.data.params.train_datasets
    action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = validate_stats(data.mean, data.std, action_length)
    return (
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.from_numpy(build_action_mask(action_length)).to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", type=str, default=None, help="For example xpu, xpu:0, cuda:0 or cpu. Defaults to xpu, then cuda, then cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = str(resolve_device(args.device))
    cfg, model = load_model(args.model, device)
    mean, std, action_mask = load_stats(cfg, device)

    try:
        server = Server(args.host, args.port, model, mean, std, action_mask, device)
        print(f"Starting server on {args.host}:{args.port}")
        # Run in this process: the model already sits on the device, and a device tensor
        # that is not a CUDA tensor (for example XPU) cannot be pickled to a spawned child.
        server.run()
    except OSError as error:
        if error.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
