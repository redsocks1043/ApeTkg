import torch
import torch.nn as nn


# 该文件实现了一个用于时间知识图谱推理的episode处理模块
# 主要包含前向推理和束搜索两种模式，用于处理动态关系路径的预测

class Episode(nn.Module):
    def __init__(self, env, agent, config):
        """初始化模块
        Args:
            env: 环境对象，提供动作空间信息
            agent: 智能体对象，包含策略网络和嵌入表示
            config: 配置参数字典
        """
        super(Episode, self).__init__()
        self.config = config  # 存储配置参数
        self.env = env  # 环境接口，用于获取可用动作
        self.agent = agent  # 智能体，包含策略网络和嵌入
        self.path_length = config['path_length']  # 推理路径的最大长度
        self.num_rel = config['num_rel']  # 关系类型的总数
        self.max_action_num = config['max_action_num']  # 每个步骤的最大候选动作数

    def forward(self, query_entities, query_timestamps, query_relations):
        """前向传播处理完整推理路径（训练用）
        Args:
            query_entities: [batch_size] 查询的起始实体ID
            query_timestamps: [batch_size] 查询的时间戳
            query_relations: [batch_size] 目标关系的ID
        Return:
            返回训练过程中各步骤的损失、logits、动作索引及最终状态
        """
        # 获取初始嵌入表示
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        # 初始化当前状态
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # 初始化NO_OP操作

        # 存储各步骤信息
        all_loss = []
        all_logits = []
        all_actions_idx = []

        # 初始化LSTM隐藏状态
        self.agent.policy_step.set_hiddenx(query_relations.shape[0])

        # 逐步执行推理路径
        for t in range(self.path_length):
            first_step = (t == 0)  # 判断是否为第一步

            # 从环境获取可用动作
            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )

            # 通过智能体选择动作
            loss, logits, action_id = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
            )

            # 提取选择的动作信息
            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(
                action_space.shape[0])

            # 保存当前步骤信息
            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)

            # 更新当前状态
            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation

        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps

    def beam_search(self, query_entities, query_timestamps, query_relations):
        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        self.agent.policy_step.set_hiddenx(batch_size)

        # 第一步
        current_entities = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        action_space = self.env.next_actions(current_entities, current_timestamps,
                                             query_timestamps, self.max_action_num, True)
        loss, logits, action_id = self.agent(
            prev_relations,
            current_entities,
            current_timestamps,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space
        )

        action_space_size = action_space.shape[1]
        beam_size = min(self.config['beam_size'], action_space_size)
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)
        beam_log_prob = beam_log_prob.reshape(-1)

        current_entities = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(-1)
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(-1)
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(-1)

        # 扩展历史编码器以适配 beam_size
        self.agent.policy_step.expand_for_beam(beam_size)

        # 后续步骤
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, beam_size).reshape(batch_size * beam_size, -1)
            query_relations_embeds_roll = query_relations_embeds.repeat(1, beam_size).reshape(batch_size * beam_size,
                                                                                              -1)

            action_space = self.env.next_actions(current_entities, current_timestamps,
                                                 query_timestamps_roll, self.max_action_num)

            loss, logits, action_id = self.agent(
                prev_relations,
                current_entities,
                current_timestamps,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space
            )

            # 更新 beam 概率
            beam_tmp = beam_log_prob.repeat(action_space_size, 1).T + logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)

            new_beam_size = min(self.config['beam_size'], action_space_size * beam_size)
            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, new_beam_size, dim=1)
            offset = top_k_action_id // action_space_size  # beam 索引

            # 更新隐藏状态
            hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            offset_expanded = offset.unsqueeze(-1).repeat(1, 1, self.agent.policy_step.state_dim)
            self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset_expanded).reshape(
                batch_size * new_beam_size, -1)
            # 更新历史序列
            history_tmp = self.agent.policy_step.history.reshape(batch_size, beam_size,
                                                                 self.agent.policy_step.history_len, -1)
            self.agent.policy_step.history = torch.gather(
                history_tmp, dim=1,
                index=offset_expanded.unsqueeze(2).repeat(1, 1, self.agent.policy_step.history_len, 1)
            ).reshape(batch_size * new_beam_size, self.agent.policy_step.history_len, self.agent.policy_step.state_dim)
            self.agent.policy_step.current_len = self.agent.policy_step.current_len.reshape(batch_size,
                                                                                            beam_size).gather(dim=1,
                                                                                                              index=offset).reshape(
                -1)

            # 更新当前状态
            current_entities = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1,
                                            index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1,
                                              index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1,
                                          index=top_k_action_id).reshape(-1)
            beam_log_prob = top_k_log_prob.reshape(-1)
            beam_size = new_beam_size

        return action_space[:, :, 1].reshape(batch_size, -1), beam_tmp