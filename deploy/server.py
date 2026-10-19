# Copyright (C) 2026 Xiaomi Corporation.
import pickle
import socket
import struct
import time
import argparse
import traceback

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from device import checkpoint_attn_argument, resolve_attn_implementation, resolve_device


class Server(mp.Process):
    def __init__(self, model_path, host, port, seed=42, device=None, attn_implementation=None):
        super(Server, self).__init__()
        self.host = host
        self.port = port

        self.model_path = model_path
        self.device = device
        self.attn_implementation = attn_implementation

    def _build_model(self):
        # Built inside the spawned process: an XPU tensor cannot be pickled to a child process.
        device = resolve_device(self.device)
        attn_implementation = resolve_attn_implementation(device, self.attn_implementation)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, attn_implementation=checkpoint_attn_argument(attn_implementation), dtype=torch.bfloat16).to(device).to(torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

    def _recv_all(self, conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def run(self):
        self._build_model()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, addr = server_socket.accept()

                try:
                    request_count = 0
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            data_len_bytes = self._recv_all(conn, 4)
                            if not data_len_bytes:
                                break
                            data_len = struct.unpack(">I", data_len_bytes)[0]

                            data = self._recv_all(conn, data_len)
                            if not data:
                                break

                            tic = time.time()

                            input_data = pickle.loads(data)
                            robot_type = input_data["task_id"]
                            data = {key: (value.to(self.model.device, self.model.dtype) if isinstance(value, torch.Tensor) else value) for key, value in input_data.items()}

                            if "bridge" in robot_type or "fractal" in robot_type:
                                instruction = data["language"]
                                vl_inputs = self.processor(
                                    text=[instruction],
                                    images=[data["base"]],
                                    videos=None,
                                    padding=True,
                                    return_tensors="pt",
                                )
                            else:
                                instruction = f"<|im_start|>user\nThe following observations are captured from multiple views.\n# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\nGenerate robot actions for the task:\n{data["language"]} /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"
                                vl_inputs = self.processor(
                                    text=[instruction],
                                    images=[data["base"], data["wrist_left"]],
                                    videos=None,
                                    padding=True,
                                    return_tensors="pt",
                                )
                            vl_inputs = vl_inputs.to(self.model.device)

                            data.update(vl_inputs)
                            data["action_mask"] = self.processor.get_action_mask(robot_type).to(self.model.device, self.model.dtype)
                            data["state"] = torch.from_numpy(data["state"]).to(self.model.device, self.model.dtype).view(1, 1, -1)

                            outputs = self.model(**data)
                            action = self.processor.decode_action(outputs.actions, robot_type=robot_type)

                            response = pickle.dumps(action.cpu())
                            conn.sendall(struct.pack(">I", len(response)) + response)

                            toc = time.time()
                            request_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "avg_time": f"{(toc - tic) * 1000:.2f}ms",
                                }
                            )

                except Exception as e:
                    print(f"Error handling connection: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model dir.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10086,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, for example xpu, xpu:0, cuda:0 or cpu. Defaults to xpu, then cuda, then cpu.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation. Defaults to flash_attention_2 on CUDA when flash_attn is installed, otherwise sdpa.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        server = Server(model_path=args.model, host=args.host, port=args.port, device=args.device, attn_implementation=args.attn_implementation)
        print(f"Starting server on {args.host}:{args.port}")
        server.start()
        server.join()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("Server interrupted")
