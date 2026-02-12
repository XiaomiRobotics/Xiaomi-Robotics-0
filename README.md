<div align="center">

  # Xiaomi-Robotics-0

  **An Open-Sourced Vision-Language-Action Model with Real-Time Inference**

  [![Paper](https://img.shields.io/badge/📄-Paper-red)](https://xiaomi-robotics-0.github.io/assets/paper.pdf)
  [![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://xiaomi-robotics-0.github.io/)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-0)
  [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## 💡 About Xiaomi-Robotics-0

**Xiaomi-Robotics-0** is a state-of-the-art **Vision-Language-Action (VLA) model** with 4.7B parameters, specifically engineered for high-performance robotic reasoning and seamless real-time execution. 

### Key Features:

* **🧠 Strong Generalization**: Pre-trained on diverse cross-embodiment trajectories and VL data to handle complex, unseen tasks.
* **🚀 Real-Time Ready**: Optimized with asynchronous execution to minimize inference latency.
* **🛠️ Flexible Deployment**: Fully compatible with the Hugging Face `transformers` ecosystem and optimized for consumer GPUs.



## 📅 Updates

- **[Feb 2026]** 🎉 Released the **Technical Report**.
- **[Feb 2026]** 🔥 Released **Pre-trained weights** and **Fine-tuned weights** for LIBERO, CALVIN, and SimplerEnv.
- **[Feb 2026]** 💻 Inference code and evaluation scripts are now live!

---


## 🏆 Benchmark

We evaluate **Xiaomi-Robotics-0** on three standard simulation benchmarks: **CALVIN**, **LIBERO**, and **SimplerEnv**. The table below summarizes the performance results across different embodiments and datasets. For each setting, we provide the corresponding fine-tuned checkpoint and a guide for running the evaluation.


|                | 🤗 Name on Hugging Face                                       | Description                       | Performance                        | Evaluation Guide                             |
| :------------- | :----------------------------------------------------------- | :-------------------------------- | :--------------------------------- | :------------------------------------------- |
| **LIBERO**     | [**Xiaomi-Robotics-0-LIBERO**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-LIBERO) | Fine-tuned on four LIBERO suites. | **98.7%** (Avg Success)            | [LIBERO Eval](eval_libero/README.md)         |
| **CALVIN**     | [**Xiaomi-Robotics-0-Calvin-ABCD_D**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D) | Fine-tuned on ABCD→D Split.       | **4.80** (Avg Length)              | [CALVIN Eval](eval_calvin/README.md)         |
|                | [**Xiaomi-Robotics-0-Calvin-ABC_D**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABC_D) | Fine-tuned on ABC→D Split.        | **4.75** (Avg Length)              | [CALVIN Eval](eval_calvin/README.md)         |
| **SimplerEnv** | [**Xiaomi-Robotics-0-SimplerEnv-Google-Robot**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot) | Fine-tuned on Fractal dataset.    | **85.5%** (VM) <br> **74.7%** (VA) | [SimplerEnv Eval](eval_simplerenv/README.md) |
|                | [**Xiaomi-Robotics-0-SimplerEnv-WidowX**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX) | Fine-tuned on Bridge dataset.     | **79.2%**                          | [SimplerEnv Eval](eval_simplerenv/README.md) |
| **Base**       | [**Xiaomi-Robotics-0**](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0) | Pre-trained model.                | -                                  | -                                            |




## 🚀 Quick Start: Installation & Deployment

Our project relies primarily on HuggingFace Transformers 🤗, making deployment extremely easy. If your environment supports transformers >= 4.57.1, you can use our project seamlessly—we recommend PyTorch 2.8.0 (paired with torchvision 0.23.0 and torchaudio 2.8.0), as this combination has been fully tested by our team and ensures optimal compatibility. 

### 1️⃣ Installation Guides

Here’s a simple installation guide to get you started:

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-0 
cd Xiaomi-Robotics-0

# install a conda env manager pixi from
# https://pixi.prefix.dev/latest/installation/
curl -fsSL https://pixi.sh/install.sh | sh

pixi install

pixi shell
```



### 2️⃣ Deployment Guides

**Xiaomi-Robotics-0** is deployed on top of the HuggingFace Transformers 🤗 ecosystem, enabling straightforward deployment for robotic manipulation tasks. By leveraging Flash Attention 2 and bfloat16 precision, the model can be loaded and run efficiently on consumer-grade GPUs.

``` python
import torch
from transformers import AutoModel, AutoProcessor

# 1. Load model and processor 
model_path = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
model = AutoModel.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    attn_implementation="flash_attention_2", 
    dtype=torch.bfloat16
).cuda().eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


# 2. Construct the prompt with multi-view inputs
language_instruction = "Pick up the red block."
instruction = (
    f"<|im_start|>user\nThe following observations are captured from multiple views.\n"
    f"# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
    f"# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
    f"Generate robot actions for the task:\n{language_instruction} /no_cot<|im_end|>\n"
    f"<|im_start|>assistant\n<cot></cot><|im_end|>\n"
)

# 3. Prepare inputs
# Assuming `image_base`, `image_wrist`, and `proprio_state` are already loaded
inputs = processor(
    text=[instruction],
    images=[image_base, image_wrist], # [PIL.Image, PIL.Image]
    videos=None,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Add proprioceptive state and action mask
robot_type = "libero"
inputs["state"] = torch.from_numpy(proprio_state).to(model.device, model.dtype).view(1, 1, -1)
inputs["action_mask"] = processor.get_action_mask(robot_type).to(model.device, model.dtype)

# 4. Generate action 
with torch.no_grad():
    outputs = model(**inputs)
    
# Decode raw outputs into actionable control commands
action_chunk = processor.decode_action(outputs.actions, robot_type=robot_type)
print(f"Generated Action Chunk Shape: {action_chunk.shape}")
```



## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@misc{robotics2026xiaomi,
  title        = {Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution},
  author       = {Xiaomi Robotics},
  howpublished={\url{https://xiaomi-robotics-0.github.io}},
  year         = {2026},
  note={Project Website}
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
