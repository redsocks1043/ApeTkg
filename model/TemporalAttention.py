import math

import torch
import torch.nn as nn
import numpy as np

class TemporalRewardAttention(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim, time_span):
        super(TemporalRewardAttention, self).__init__()

        self.num_heads = 8  # 2025.3.11 注意力头由16->8减轻复杂度
        self.hidden_dim = (hidden_dim // self.num_heads) * self.num_heads

        # 实体和关系的嵌入
        self.entity_embeddings = nn.Embedding(num_entities, self.hidden_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.hidden_dim)
        self.time_embeddings = nn.Embedding(time_span, self.hidden_dim)


        # 改进时间编码-使用周期性和线性组合的时间编码
        self.time_encoder = TimeEncoder(self.hidden_dim)
        # 位置编码
        self.position_encoding = self.create_position_encoding(time_span, self.hidden_dim)

        # 添加查询变换
        self.query_transform = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        # 添加时间门控
        self.time_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)


        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim) # 对融合后的实体和关系特征归一化
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim) # 自注意力后的归一化

        # 特征变换
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)   # 降低dropout比例减少过拟合
        )

        # 多头注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def create_position_encoding(max_len, d_model):
        # 改进的位置编码
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_encoding, requires_grad=False)

    def forward(self, quadruples):
        # 获取嵌入
        subj = self.entity_embeddings(quadruples[:, 0])
        rel = self.relation_embeddings(quadruples[:, 1])
        obj = self.entity_embeddings(quadruples[:, 2])
        # time = self.time_embeddings(quadruples[:, 3])

        # 添加：使用改进的时间编码
        time_encoding = self.time_encoder(quadruples[:, 3])
        time = self.time_embeddings(quadruples[:, 3]) + time_encoding

        # # 提高注意力
        # query = subj + obj
        # key = rel
        # value = rel
        '''添加部分开始'''
        # 将时间信息融入实体表示
        subj = subj + time
        obj = obj + time
        # 提高注意力 - 修改注意力计算方式
        # 使用实体和关系的交互作为查询
        query = torch.cat([subj, rel, obj], dim=-1)
        query = self.query_transform(query)

        # 使用时间作为键和值
        key = time
        value = time
        '''添加部分结束'''

        # 三元组特征融合
        combined = torch.cat([subj, rel, obj], dim=-1)
        transformed = self.transform(combined)

        # 层归一化
        transformed = self.layer_norm1(transformed)

        # # 自注意力机制
        # attn_output, _ = self.self_attention(query.unsqueeze(1),key.unsqueeze(1),value.unsqueeze(1))
        # 自注意力机制 - 修改为多头交叉注意力
        attn_output, _ = self.self_attention(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)

        # 残差连接和层归一化
        attn_output = self.layer_norm2(transformed + attn_output)

        # 时序注意力
        # temporal_output = attn_output * time
        # 修改：时序注意力 - 改为门控机制
        # 简化的时间门控
        gate = torch.sigmoid(self.time_gate(torch.cat([attn_output, time], dim=-1)))
        temporal_output = gate * attn_output + (1 - gate) * transformed  # 使用transformed而不是time
        # 前馈网络
        ff_output = self.feed_forward(temporal_output)
        ff_output = self.layer_norm2(temporal_output + ff_output)

        # 输出奖励值
        rewards = self.output_layer(ff_output)

        return rewards.squeeze(-1)


class TimeEncoder(nn.Module):
    """简化的时间编码器"""

    def __init__(self, hidden_dim):
        super(TimeEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # 简化为单一线性映射
        self.time_linear = nn.Linear(1, hidden_dim)
        self.activation = nn.GELU()
        # 添加归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, time_idx):
        """
        Args:
            time_idx: 时间索引 [batch_size]
        Returns:
            时间编码 [batch_size, hidden_dim]
        """
        # 线性时间特征
        time_linear = self.time_linear(time_idx.float().view(-1, 1))
        return self.norm(self.activation(time_linear))

class TemporalRewardLearner:
    """时序奖励学习器"""
    def __init__(self, quadruples, num_entities, num_relations, hidden_dim, time_span,
                 device='cuda', lr=0.001, batch_size=128, num_epochs=100, label_smoothing=0.1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")  # 添加设备信息打印
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.time_span = time_span
        self.label_smoothing = label_smoothing  # 添加标签平滑参数

        # 添加正则化
        self.l2_lambda = 0.001  # 降低
        # 简化时间权重
        self.time_weights = torch.exp(-torch.arange(time_span, dtype=torch.float32) / time_span).to(self.device)
        
        # 数据预处理：简化时间戳处理
        quadruples = np.array(quadruples)
        # 标准化时间戳 - 使用简单的线性映射
        max_time = np.max(quadruples[:, 3])
        normalized_times = (quadruples[:, 3] * (time_span - 1) / max_time).astype(int)
        quadruples = quadruples.copy()
        quadruples[:, 3] = normalized_times
        # 将数据转换为tensor但保留在CPU上
        self.quadruples = torch.LongTensor(quadruples)  # 不要使用.to(self.device)

        # 初始化模型
        self.model = TemporalRewardAttention(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            time_span=time_span
        ).to(self.device)

        # 优化器->2025.3.2 具有权重衰减和学习率调度器的优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # 使用更平缓的学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练模型
        self.rewards = self.train_model()

    def compute_reward_target(self, batch_quads):
        """
        计算更简单但有效的目标奖励，并应用标签平滑
        不再使用时间衰减，而是将所有时间点视为同等重要
        """
        # 关系频率 - 简化计算
        rel_counts = torch.bincount(batch_quads[:, 1], minlength=self.model.relation_embeddings.num_embeddings)
        rel_freq = rel_counts[batch_quads[:, 1]] / (rel_counts.sum() + 1e-8)

        # 头实体频率
        subj_counts = torch.bincount(batch_quads[:, 0], minlength=self.model.entity_embeddings.num_embeddings)
        subj_freq = subj_counts[batch_quads[:, 0]] / (subj_counts.sum() + 1e-8)

        # 简化的奖励计算 - 不再使用时间衰减
        raw_rewards = 0.5 * rel_freq + 0.5 * subj_freq
        
        # 应用标签平滑
        smoothed_rewards = raw_rewards * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        return smoothed_rewards.to(self.device)

    def train_model(self):
        """训练模型并返回学习到的奖励"""
        self.model.train()
        num_samples = len(self.quadruples)
        best_loss = float('inf')
        patience = 15  # 增加耐心值
        patience_counter = 0
    
        # 添加梯度累积
        accumulation_steps = 4  # 每4步更新一次参数
        
        # 添加对比损失权重
        contrastive_weight = 0.2
        
        # 使用DataLoader提高数据加载效率
        dataset = torch.utils.data.TensorDataset(self.quadruples)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,  
            num_workers=0     # 使用0个工作进程加载数据
        )
    
        # 计算总批次数，用于进度条显示
        total_batches = len(dataloader)
        
        # 创建进度条装饰器
        from tqdm import tqdm
        
        print(f"开始训练，共 {self.num_epochs} 个 epoch，每个 epoch 包含 {total_batches} 个批次")
    
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # 创建当前epoch的进度条
            progress_bar = tqdm(dataloader, 
                           desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                           leave=True, 
                           ncols=100, 
                           colour='green',
                           unit='batch')
    
            self.optimizer.zero_grad()  # 在epoch开始时清零梯度
    
            for batch in progress_bar:
                batch_quads = batch[0].to(self.device)  # 将批次数据移至GPU
                
                # 前向传播
                predicted_rewards = self.model(batch_quads)
                target_rewards = self.compute_reward_target(batch_quads)
    
                # 计算主损失
                main_loss = self.criterion(predicted_rewards, target_rewards)
                
                # 添加对比损失 - 鼓励相似四元组有相似奖励
                if len(batch_quads) > 1:
                    # 计算四元组之间的相似度
                    subj_sim = (batch_quads[:, 0].unsqueeze(1) == batch_quads[:, 0].unsqueeze(0)).float()
                    rel_sim = (batch_quads[:, 1].unsqueeze(1) == batch_quads[:, 1].unsqueeze(0)).float()
                    obj_sim = (batch_quads[:, 2].unsqueeze(1) == batch_quads[:, 2].unsqueeze(0)).float()
                    time_diff = torch.abs(batch_quads[:, 3].unsqueeze(1) - batch_quads[:, 3].unsqueeze(0)).float()
                    time_sim = torch.exp(-time_diff / self.time_span)
                    
                    # 综合相似度
                    quad_sim = (subj_sim + rel_sim + obj_sim + time_sim) / 4.0
                    
                    # 计算预测奖励之间的差异
                    reward_diff = torch.abs(predicted_rewards.unsqueeze(1) - predicted_rewards.unsqueeze(0))
                    
                    # 对比损失：相似四元组应有相似奖励
                    contrastive_loss = torch.mean(quad_sim * reward_diff)
                    
                    # 组合损失
                    combined_loss = main_loss + contrastive_weight * contrastive_loss
                else:
                    combined_loss = main_loss
    
                # L2正则化 - 简化计算
                l2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
    
                # 总损失
                total_loss = combined_loss + self.l2_lambda * l2_reg
    
                # 缩放损失以适应梯度累积
                total_loss = total_loss / accumulation_steps
    
                # 反向传播
                total_loss.backward()
                
                # 梯度累积
                if (num_batches + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
    
                epoch_loss += total_loss.item() * accumulation_steps  # 恢复原始损失大小
                num_batches += 1
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    'loss': f'{total_loss.item() * accumulation_steps:.6f}',
                    'avg_loss': f'{epoch_loss/num_batches:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.8f}'
                })
    
            # 处理最后一批可能不完整的梯度
            if num_batches % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
    
            avg_epoch_loss = epoch_loss / num_batches
    
            # 使用新的学习率调度器
            self.scheduler.step(avg_epoch_loss)
    
            # 早停
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
    
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                  f'Loss: {avg_epoch_loss:.6f}, '
                  f'Best Loss: {best_loss:.6f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
    
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # 恢复最佳模型
                self.model.load_state_dict(best_model_state)
                self.model = self.model.to(self.device)
                break
    
        # 模型评估
        # 模型评估
        self.model.eval()
        with torch.no_grad():
            # 使用更大的批次进行评估
            eval_batch_size = self.batch_size * 2
            all_rewards = []
            
            # 创建评估数据加载器
            eval_dataset = torch.utils.data.TensorDataset(self.quadruples)
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, 
                batch_size=eval_batch_size,
                shuffle=False,
                pin_memory=False,  # 禁用pin_memory
                num_workers=0
            )
            
            for batch in eval_dataloader:
                batch_quads = batch[0].to(self.device)  # 将批次数据移至GPU
                rewards = self.model(batch_quads)
                all_rewards.append(rewards.cpu())
                
            return torch.cat(all_rewards).numpy()


class AttentionRewardDistribution:
    def __init__(self, rewards, device):
        """
        Args:
            rewards: 注意力模型计算的奖励值
            device: 计算设备(CPU/GPU)
        """
        self.rewards = torch.FloatTensor(rewards).to(device)
        self.device = device

    def sample(self, indices=None):
        """获取奖励值
        Args:
            indices: 需要获取奖励的四元组索引
        Returns:
            相应的奖励值
        """
        if indices is None:
            return self.rewards
        return self.rewards[indices]

    def get_reward(self, state):
        """
        获取特定状态的奖励值
        Args:
            state: 当前状态，包含实体和关系信息
        Returns:
            对应的奖励值
        """
        # 获取状态中的实体和关系索引
        entity_idx = state.get('entity_index', 0)
        reward = self.rewards[entity_idx]
        return torch.tensor(reward, device=self.device)

    def __call__(self, rel, time_step):
        # 根据关系调整奖励，不再考虑时间步长
        idx = rel  # 关系索引
        base_reward = self.rewards[idx]
    
        # 不再使用时间因子进行衰减
        final_reward = base_reward
        return final_reward


class DynamicRewardAdjustment(nn.Module):
    """动态奖励调整模块，用于计算奖励增量ΔR"""
    def __init__(self, num_entities, num_relations, hidden_dim, device='cuda'):
        super(DynamicRewardAdjustment, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 实体和关系的嵌入
        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 奖励调整网络
        self.reward_adjustment = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 使用Tanh激活函数，输出范围为[-1, 1]
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, current_entities, relations, target_entities, base_rewards):
        """
        计算奖励增量ΔR
        Args:
            current_entities: 当前实体 [batch_size] 或标量
            relations: 关系 [batch_size] 或标量
            target_entities: 目标实体 [batch_size] 或标量
            base_rewards: 基础奖励R [batch_size] 或标量
        Returns:
            reward_delta: 奖励增量ΔR [batch_size]
        """
        # 确保输入是张量
        if not isinstance(current_entities, torch.Tensor):
            current_entities = torch.tensor([current_entities], device=self.device)
        if not isinstance(relations, torch.Tensor):
            relations = torch.tensor([relations], device=self.device)
        if not isinstance(target_entities, torch.Tensor):
            target_entities = torch.tensor([target_entities], device=self.device)
        if not isinstance(base_rewards, torch.Tensor):
            base_rewards = torch.tensor([base_rewards], device=self.device)
        
        # 确保张量维度正确
        if current_entities.dim() == 0:
            current_entities = current_entities.unsqueeze(0)
        if relations.dim() == 0:
            relations = relations.unsqueeze(0)
        if target_entities.dim() == 0:
            target_entities = target_entities.unsqueeze(0)
        if base_rewards.dim() == 0:
            base_rewards = base_rewards.unsqueeze(0)
        
        # 获取嵌入
        current_emb = self.entity_embeddings(current_entities)
        relation_emb = self.relation_embeddings(relations)
        target_emb = self.entity_embeddings(target_entities)
        
        # 编码当前状态
        state_input = torch.cat([current_emb, relation_emb, target_emb], dim=-1)
        state_encoding = self.state_encoder(state_input)
        
        # 结合基础奖励和状态编码
        reward_input = torch.cat([state_encoding, base_rewards.unsqueeze(-1).expand(-1, self.hidden_dim)], dim=-1)
        
        # 计算奖励增量
        reward_delta = self.reward_adjustment(reward_input)
        
        # 缩放奖励增量，使其不会过度影响基础奖励
        return reward_delta.squeeze(-1) * 0.2  # 初始缩放因子为0.2
        
    def update_scale_factor(self, epoch, max_epochs, max_scale=0.5):
        """随着训练进行，逐渐增加奖励增量的影响"""
        return min(0.2 + (max_scale - 0.2) * epoch / max_epochs, max_scale)


class CompoundRewardDistribution:
    """复合奖励分布，结合基础奖励和动态奖励增量"""
    def __init__(self, base_distribution, reward_adjustment, device):
        """
        Args:
            base_distribution: 基础奖励分布
            reward_adjustment: 动态奖励调整模块
            device: 计算设备
        """
        self.base_distribution = base_distribution
        self.reward_adjustment = reward_adjustment
        self.device = device
        self.scale_factor = 0.2  # 初始缩放因子
        
    def update_scale_factor(self, epoch, max_epochs):
        """更新缩放因子"""
        self.scale_factor = self.reward_adjustment.update_scale_factor(epoch, max_epochs)
        
    def __call__(self, rel, dt, current_entity=None, target_entity=None):
        """
        计算复合奖励
        Args:
            rel: 关系ID
            dt: 时间差
            current_entity: 当前实体
            target_entity: 目标实体
        Returns:
            复合奖励值
        """
        # 获取基础奖励
        base_reward = self.base_distribution(rel, dt)
        
        # 如果没有提供实体信息，只返回基础奖励
        if current_entity is None or target_entity is None:
            return base_reward
        
        # 计算奖励增量
        with torch.no_grad():
            # 确保关系ID是张量
            rel_tensor = torch.tensor([rel], device=self.device)
            reward_delta = self.reward_adjustment(
                current_entity.unsqueeze(0),
                rel_tensor,
                target_entity.unsqueeze(0),
                torch.tensor([base_reward], device=self.device)
            )
        
        # 组合奖励
        compound_reward = base_reward + reward_delta.item() * self.scale_factor
        
        # 确保奖励值在合理范围内
        return max(min(compound_reward, 1.0), 0.0)