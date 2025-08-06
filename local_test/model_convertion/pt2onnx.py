# import torch
# from ultralytics import YOLO

# # 1) Load the ultralytics package just to get the weights &
# #    the raw nn.Module.  Don't use its .forward/predict path.
# y = YOLO("./../models/shapes_model0719.pt")
# core = y.model.model         # <-- this is a plain nn.Sequential/backbone+head
# core.eval().cpu()           # eval mode + move to GPU

# # 2) Wrap it in a simple one‑input Module to ensure forward(x) only takes one tensor.
# class CoreWrap(torch.nn.Module):
#     def __init__(self, m):
#         super().__init__()
#         self.m = m
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.m(x)

# wrapped = CoreWrap(core)

# # 3) Create a dummy input matching your intended inference size:
# #    if you trained on 640×640:
# example = torch.zeros(1,3,640,640)

# # 4) Trace it, non‑strict to allow simple control‐flow variants:
# traced = torch.jit.trace(wrapped, example, strict=False)

# # 5) Save out your traced module:
# torch.jit.save(traced, "yolov8_core_traced.pt")


# import torch
# from ultralytics import YOLO

# # 1. 取到原始模型
# orig_model = YOLO('./../models/shapes_model0719.pt').model
# orig_model.eval()

# # 2. 定义一个只接受固定输入签名的 wrapper
# class CleanWrapper(torch.nn.Module):
#     def __init__(self, m):
#         super().__init__()
#         self.m = m
#     # 只接受一个参数 x，没有 *args **kwargs
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.m(x)

# wrapper = CleanWrapper(orig_model)

# # 3. Script 它
# scripted = torch.jit.script(wrapper)
# torch.jit.save(scripted, 'yolov8_clean_scripted.pt')


# import torch
# from ultralytics import YOLO

# # ——— 1) 取出“纯网络”：backbone + head ———
# yolo = YOLO('./../models/shapes_model0719.pt')  # 载入 ultralytics YOLO 对象
# core_net = yolo.model.model                    # 注意这里多了一个 .model

# # 切到 eval 模式（script 前不必搬设备）
# core_net.eval()

# # ——— 2) 用 wrapper 包一下，forward 只收一个 Tensor ———
# class CoreWrapper(torch.nn.Module):
#     def __init__(self, net):
#         super().__init__()
#         self.net = net
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

# wrapped = CoreWrapper(core_net)

# # ——— 3) Script 它 ———
# scripted = torch.jit.script(wrapped)            # 这次不会再去编 ultralytics/task 里的 forward
# torch.jit.save(scripted, 'yolov8_core_scripted.pt')



# import torch
# from ultralytics import YOLO
# from typing import Tuple  # <-- 导入

# # 1) 取出纯网络
# yolo = YOLO('./../models/shapes_model0719.pt')
# core_net = yolo.model.model
# core_net.eval()

# # 2) 用带返回类型注解的 wrapper 包装
# class CoreWrapper(torch.nn.Module):
#     def __init__(self, net: torch.nn.Module):
#         super().__init__()
#         self.net = net

#     # 假设 core_net(x) 返回 (Tensor, Tensor, Tensor, Tensor)
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
#         return self.net(x)  # 返回的 tuple 元素个数、类型要和这里注解一致

# wrapped = CoreWrapper(core_net)

# # 3) 脚本化并保存
# scripted = torch.jit.script(wrapped)
# torch.jit.save(scripted, 'yolov8_core_scripted.pt')



# import torch
# from ultralytics import YOLO

# y = YOLO('./../models/shapes_model0719.pt').model.model.eval()
# out = y(torch.zeros(1,3,640,640))   # 记得根据你训练的输入尺寸改
# print(type(out), len(out))          # 例如：<class 'tuple'> 4

from ultralytics import YOLO

# 1) 加载你的训练好 .pt
model = YOLO('./../models/shapes_model0719.pt')
model.model.fuse()  # 融合BN卷积
model.model.eval()

# 2) 导出到 ONNX（batch=1, static 图, opset12, 简化）
model.export(format='onnx',
             opset=12,
             dynamic=False,
             simplify=True,
             imgsz=640,
             )
