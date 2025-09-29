import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class HistoryEncoder(nn.Module):
#     def __init__(self, config):
#         super(HistoryEncoder, self).__init__()
#         self.config = config
#         # 使用LSTM单元，输入维度为action_dim，输出维度为state_dim
#         self.lstm_cell = torch.nn.LSTMCell(input_size=config['action_dim'],
#                                            hidden_size=config['state_dim'])
#
#     def set_hiddenx(self, batch_size):
#         """初始化隐藏层参数，设为0"""
#         if self.config['cuda']:
#             self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
#             self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
#         else:
#             self.hx = torch.zeros(batch_size, self.config['state_dim'])
#             self.cx = torch.zeros(batch_size, self.config['state_dim'])
#
#     def forward(self, prev_action, mask):
#         """mask: 如果是NO_OP（无操作），则该位置不更新历史编码结果"""
#         # 将上一时刻的动作传入LSTM单元，得到当前隐藏状态和细胞状态
#         self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
#         # 如果是NO_OP，则保持历史状态不变X
#         self.hx = torch.where(mask, self.hx, self.hx_)
#         self.cx = torch.where(mask, self.cx, self.cx_)
#         return self.hx


class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.state_dim = config['state_dim']
        self.history_len = config.get('history_len', 10)
        self.input_proj = nn.Linear(config['action_dim'], self.state_dim)
        num_heads = next(num for num in [8, 4, 2, 1] if self.state_dim % num == 0)
        self.self_attn = nn.MultiheadAttention(embed_dim=self.state_dim, num_heads=num_heads, batch_first=True)

    def set_hiddenx(self, batch_size):
        device = 'cuda' if self.config['cuda'] else 'cpu'
        self.history = torch.zeros(batch_size, self.history_len, self.state_dim, device=device)
        self.current_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.hx = torch.zeros(batch_size, self.state_dim, device=device)
        # 移除 self.cx，因为自注意力机制不需要它

    def forward(self, prev_action, mask):
        batch_size = prev_action.shape[0]
        if mask.dim() > 1 and mask.size(1) > 1:
            mask = mask[:, 0]

        action_proj = self.input_proj(prev_action).unsqueeze(1)  # [batch_size, 1, state_dim]
        new_history = torch.cat([action_proj, self.history[:, :-1]], dim=1)
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        self.history = torch.where(mask_expanded, self.history, new_history)

        self.current_len = torch.where(
            mask,
            self.current_len,
            torch.clamp(self.current_len + 1, max=self.history_len)
        )

        attn_mask = torch.arange(self.history_len, device=prev_action.device)[None, :] >= self.current_len[:, None]
        attn_output, _ = self.self_attn(self.history, self.history, self.history, key_padding_mask=attn_mask)
        self.hx_ = attn_output[:, 0]
        self.hx = torch.where(mask.unsqueeze(1), self.hx, self.hx_)
        return self.hx

    def expand_for_beam(self, beam_size):
        batch_size = self.history.shape[0]
        # 扩展历史序列、长度计数器和隐藏状态
        self.history = self.history.repeat(beam_size, 1, 1).reshape(batch_size * beam_size, self.history_len, self.state_dim)
        self.current_len = self.current_len.repeat(beam_size)
        self.hx = self.hx.repeat(beam_size, 1)

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        # 定义一个两层的MLP
        self.mlp_l1 = nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)

    def forward(self, state_query):
        """将状态查询传入MLP网络，得到动作概率输出"""
        hidden = torch.relu(self.mlp_l1(state_query))  # 第一层MLP
        output = self.mlp_l2(hidden).unsqueeze(1)  # 第二层MLP
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super(DynamicEmbedding, self).__init__()
        # 实体嵌入：动态嵌入方法
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())  # 时间相关权重
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())  # 时间偏置

    def forward(self, entities, dt):
        """根据时间差（dt）和实体ID获取动态嵌入"""
        dt = dt.unsqueeze(-1)  # 扩展维度，使其适应参数w和b
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))  # 时间嵌入
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)  # 获取实体嵌入
        return torch.cat((e, t), -1)  # 合并实体嵌入和时间嵌入
class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        # 实体嵌入：静态嵌入方法
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        """获取实体的静态嵌入"""
        return self.ent_embs(entities)

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2  # 计算总关系数，包括正向和反向关系，以及无操作（stay in place）
        self.config = config

        # 定义特殊标记：
        # [0, num_rel) -> 正常关系；
        # num_rel -> 保持不动（无操作）；(num_rel, num_rel * 2] -> 反向关系
        self.NO_OP = self.num_rel  # 无操作（保持不动）
        self.ePAD = config['num_ent']  # 实体填充标记
        self.rPAD = config['num_rel'] * 2 + 1  # 关系填充标记
        self.tPAD = 0  # 时间填充标记

        # 根据配置选择实体嵌入方式：动态嵌入或静态嵌入
        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'])
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])

        # 关系嵌入
        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])
        # 历史编码器（LSTM）->2025.3.29修改为自注意力机制
        self.policy_step = HistoryEncoder(config)
        # 策略MLP
        self.policy_mlp = PolicyMLP(config)
        # 分数加权全连接层，用于计算不同动作的加权分数
        self.score_weighted_fc = nn.Linear(
            self.config['ent_dim'] * 2 + self.config['rel_dim'] * 2 + self.config['state_dim'],
            1, bias=True)

    def forward(self, prev_relation, current_entities, current_timestamps,
                query_relation, query_entity, query_timestamps, action_space):
        """
        前向传播：
        参数：
            prev_relation: [batch_size]  上一时刻的关系
            current_entities: [batch_size] 当前的实体
            current_timestamps: [batch_size] 当前时间戳
            query_relation: [batch_size, rel_dim] 查询关系的嵌入
            query_entity: [batch_size, ent_dim] 查询实体的嵌入
            query_timestamps: [batch_size] 查询时间戳
            action_space: [batch_size, max_actions_num, 3] 每个动作包含关系、实体和时间戳
        """
        # 计算当前时刻和查询时刻的时间差
        current_delta_time = query_timestamps - current_timestamps
        # 获取当前实体的嵌入
        current_embds = self.ent_embs(current_entities, current_delta_time)  # [batch_size, ent_dim]
        # 获取上一时刻关系的嵌入
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # 创建填充mask，用于忽略填充的关系
        pad_mask = torch.ones_like(action_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]

        # 历史状态编码：计算上一时刻的LSTM状态
        # 如果是NO_OP（无操作），则在状态维度上保持不变
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        # 将关系嵌入和实体嵌入拼接
        prev_action_embedding = torch.cat([prev_relation_embds, current_embds], dim=-1)  # [batch_size, rel_dim + ent_dim]
        # 输入LSTM计算历史编码
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim]

        # 邻居/候选动作的嵌入
        action_num = action_space.size(1)
        # 计算与每个候选动作的时间差
        neighbors_delta_time = query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        # 获取候选动作的实体嵌入
        neighbors_entities = self.ent_embs(action_space[:, :, 1], neighbors_delta_time)  # [batch_size, action_num, ent_dim]
        # 获取候选动作的关系嵌入
        neighbors_relations = self.rel_embs(action_space[:, :, 0])  # [batch_size, action_num, rel_dim]

        # 代理状态表示，将LSTM输出、查询实体和查询关系拼接起来
        agent_state = torch.cat([lstm_output, query_entity, query_relation], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim]
        # 将代理状态传入MLP网络，得到各个动作的分数输出
        output = self.policy_mlp(agent_state)  # [batch_size, 1, action_dim]  action_dim == rel_dim + ent_dim

        # 计算得分
        # 提取出与实体相关的输出
        entitis_output = output[:, :, self.config['rel_dim']:]  # [batch_size, 1, ent_dim]
        # 提取出与关系相关的输出
        relation_ouput = output[:, :, :self.config['rel_dim']]  # [batch_size, 1, rel_dim]
        # 计算关系得分：关系嵌入与候选关系嵌入的点积
        relation_score = torch.sum(torch.mul(neighbors_relations, relation_ouput), dim=2)
        # 计算实体得分：实体嵌入与候选实体嵌入的点积
        entities_score = torch.sum(torch.mul(neighbors_entities, entitis_output), dim=2)  # [batch_size, action_number]

        # 拼接关系嵌入和实体嵌入，作为动作的完整表示
        actions = torch.cat([neighbors_relations, neighbors_entities], dim=-1)  # [batch_size, action_number, action_dim]

        # 将代理状态复制到所有动作，准备输入加权分数计算
        agent_state_repeats = agent_state.unsqueeze(1).repeat(1, actions.shape[1], 1)
        # 拼接动作和代理状态，作为输入
        score_attention_input = torch.cat([actions, agent_state_repeats], dim=-1)
        # 使用加权全连接层计算注意力分数
        a = self.score_weighted_fc(score_attention_input)
        a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]，注意力分数

        # 根据注意力分数加权计算最终的得分
        scores = (1 - a) * relation_score + a * entities_score  # [batch_size, action_number]

        # 使用pad_mask将填充部分的得分置为负无穷
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        # 使用softmax计算每个动作的概率
        action_prob = torch.softmax(scores, dim=1)
        # 根据概率随机选择一个动作
        action_id = torch.multinomial(action_prob, 1)  # 随机选择一个动作。 [batch_size, 1]

        # 计算log_softmax作为损失函数的输入
        logits = torch.nn.functional.log_softmax(scores, dim=1)  # [batch_size, action_number]
        # 将选择的动作位置标记为1，其余位置为0，生成one-hot编码
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        # 计算负对数似然损失
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)
        return loss, logits, action_id

    def get_im_embedding(self, cooccurrence_entities):
        """获取共现关系的归纳平均表示（Inductive Mean Representation）。
        参数：
            cooccurrence_entities: 包含与共现关系相关的训练实体的列表。
        返回：
            torch.tensor: 共现实体的表示。
        """
        # 获取共现实体的嵌入
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        # 计算这些实体的平均嵌入表示
        im = torch.mean(entities, dim=0)
        return im

    def update_entity_embedding(self, entity, ims, mu):
        """根据共现关系更新实体的表示。
        参数：
            entity: int，需要更新的实体的索引。
            ims: torch.tensor，[共现数目, -1]，共现关系的IM表示。
            mu: 更新比率，超参数。
        """
        # 保存原始实体嵌入
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        # 使用IM表示和超参数mu更新实体的嵌入
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """进行实体嵌入的预测偏移（shift）。"""
        # 保存实体的原始嵌入
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        # 使用IM表示和超参数mu进行偏移更新
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * im

    def back_entities_embedding(self, entity):
        """在偏移结束后，恢复实体的原始嵌入。"""
        # 恢复实体的原始嵌入
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity

