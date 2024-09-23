import torch
import torch.nn.functional as F

# 创建一个16*32*32的随机张量
input_tensor = torch.randn(1, 16, 32, 32)  # 假设有一个批量大小和16个通道

# 最大池化下采样
maxpooled_tensor = F.max_pool2d(input_tensor, kernel_size=2, stride=2)

# 平均池化下采样
avgpooled_tensor = F.avg_pool2d(input_tensor, kernel_size=2, stride=2)

# 自适应平均池化下采样
adaptive_avgpooled_tensor = F.adaptive_avg_pool2d(input_tensor, output_size=(8, 16))

print("Original shape:", input_tensor.shape)
print("Max pooled shape:", maxpooled_tensor.shape)
print("Average pooled shape:", avgpooled_tensor.shape)
print("Adaptive average pooled shape:", adaptive_avgpooled_tensor.shape)