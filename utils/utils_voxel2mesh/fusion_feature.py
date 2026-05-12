import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureFusionTransformer(nn.Module):
    def __init__(self, transformer_dim, transformer_heads=8):
        super(FeatureFusionTransformer, self).__init__()

        self.transformer_dim = transformer_dim

        # 分别转换每个特征到相同的维度
        # self.feature_transforms = nn.ModuleList([
        #     nn.Linear(feat_dim, transformer_dim) for feat_dim in feature_dims
        # ])

        # Transformer的自注意力部分
        self.multi_head_attn = nn.MultiheadAttention(transformer_dim, transformer_heads)

        # 输出转换层（如果需要）
        self.output_transform = nn.Linear(transformer_dim, transformer_dim)

    def forward(self, feature_list):
        # 假设feature_list是一个列表，其中每个元素的尺寸为[N, num_vertices, feature_dim]
        # transformed_features = []

        # 对每个特征应用独立的转换
        # for i, feature in enumerate(feature_list):
        #     transformed = self.feature_transforms[i](feature)
        #     transformed_features.append(transformed)

        # 将特征堆叠在一起，准备进行自注意力操作
        stacked_features = torch.cat(feature_list, dim=0)

        # 注意力模型期待的输入是 [L, N, C]，因此需要调整特征的顺序
        attn_input = stacked_features.permute(1, 0, 2)

        # 应用自注意力
        attn_output, _ = self.multi_head_attn(attn_input, attn_input, attn_input)

        # 对输出进行进一步处理（如果需要）
        output = self.output_transform(attn_output.permute(1, 0, 2))

        output = output.mean(dim=0)[None]

        return output