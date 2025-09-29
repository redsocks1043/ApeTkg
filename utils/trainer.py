import torch  # 导入PyTorch库
import json  # 导入JSON处理库
import os  # 导入操作系统功能库
import tqdm  # 导入进度条库

class Trainer(object):  # 定义训练器类
    def __init__(self, model, pg, optimizer, args, distribution=None):  # 初始化方法，接收模型、策略梯度、优化器、参数和分布
        self.model = model  # 保存模型实例
        self.pg = pg  # 保存策略梯度实例
        self.optimizer = optimizer  # 保存优化器实例
        self.args = args  # 保存参数
        self.distribution = distribution  # 保存分布实例，用于奖励整形
        self.compound_reward = hasattr(args, 'compound_reward') and args.compound_reward  # 是否使用复合奖励
        self.reward_cache = {}  # 添加奖励缓存
        self.debug = hasattr(args, 'debug') and args.debug  # 添加调试标志
        # 添加设备属性
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def train_epoch(self, dataloader, ntriple):  # 训练一个epoch的方法
        self.model.train()  # 将模型设置为训练模式
        # 如果使用复合奖励，将奖励调整模块设置为训练模式
        if self.compound_reward and hasattr(self.distribution, 'reward_adjustment'):
            self.distribution.reward_adjustment.train()
            
        total_loss = 0.0  # 初始化总损失
        total_reward = 0.0  # 初始化总奖励
        counter = 0  # 初始化计数器
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:  # 创建进度条，总数为ntriple，单位为'ex'
            bar.set_description('Train')  # 设置进度条描述为'Train'
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:  # 遍历数据加载器中的批次数据
                if self.args.cuda:  # 如果启用了CUDA
                    src_batch = src_batch.cuda()  # 将源实体批次数据移至GPU
                    rel_batch = rel_batch.cuda()  # 将关系批次数据移至GPU
                    dst_batch = dst_batch.cuda()  # 将目标实体批次数据移至GPU
                    time_batch = time_batch.cuda()  # 将时间批次数据移至GPU

                all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)  # 通过模型前向传播获取损失、logits、当前实体和当前时间

                reward = self.pg.get_reward(current_entities, dst_batch)  # 获取当前实体和目标实体之间的奖励
                if self.args.reward_shaping:  # 如果启用了奖励整形
                    # reward shaping  # 奖励整形
                    delta_time = time_batch - current_time  # 计算时间差
                    
                    # 一次性将所有数据移至CPU进行处理
                    if self.compound_reward:
                        # 在CPU上处理实体信息和奖励计算
                        rel_cpu = rel_batch.cpu()
                        dt_cpu = (delta_time // self.args.time_span).cpu()
                        current_entities_cpu = current_entities.cpu()
                        dst_batch_cpu = dst_batch.cpu()
                        
                        # 计算奖励
                        p_dt = self._calculate_rewards_on_cpu(rel_cpu, dt_cpu, current_entities_cpu, dst_batch_cpu)
                        p_dt = p_dt.to(self.device)  # 一次性移回GPU
                    else:
                        # 标准奖励计算
                        p_dt = []
                        for i in range(rel_batch.shape[0]):
                            rel = rel_batch[i].item()
                            dt = delta_time[i].item() // self.args.time_span
                            p_dt.append(self.distribution(rel, dt))
                        p_dt = torch.tensor(p_dt, device=self.device)  # 直接在GPU上创建张量
                    
                    shaped_reward = (1 + p_dt) * reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:  # 如果未启用奖励整形
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)  # 直接计算累积折扣奖励
                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)  # 计算强化学习损失
                self.pg.baseline.update(torch.mean(cum_discounted_reward))  # 更新基线值
                self.pg.now_epoch += 1  # 当前epoch计数加1

                self.optimizer.zero_grad()  # 清空优化器梯度
                reinfore_loss.backward()  # 反向传播计算梯度
                if self.args.clip_gradient:  # 如果启用了梯度裁剪
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)  # 裁剪梯度
                self.optimizer.step()  # 优化器更新参数

                total_loss += reinfore_loss  # 累加损失
                total_reward += torch.mean(reward)  # 累加平均奖励
                counter += 1  # 计数器加1
                bar.update(self.args.batch_size)  # 更新进度条，步长为batch_size
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())  # 在进度条后显示当前损失和奖励
        return total_loss / counter, total_reward / counter  # 返回平均损失和平均奖励

    def save_model(self, checkpoint_path='checkpoint.pth'):  # 保存模型方法，默认保存路径为'checkpoint.pth'
        """Save the parameters of the model and the optimizer,"""  # 保存模型和优化器的参数
        argparse_dict = vars(self.args)  # 将参数转换为字典
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:  # 打开配置文件用于写入
            json.dump(argparse_dict, fjson)  # 将参数字典保存为JSON文件

        # 保存模型和奖励调整模块（如果存在）
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        # 如果使用复合奖励，保存奖励调整模块的状态
        if self.compound_reward and hasattr(self.distribution, 'reward_adjustment'):
            save_dict['reward_adjustment_state_dict'] = self.distribution.reward_adjustment.state_dict()
            
        torch.save(save_dict, os.path.join(self.args.save_path, checkpoint_path))


    def _calculate_rewards_on_cpu(self, rel_batch, dt_batch, current_entities, dst_batch):
        """在CPU上计算奖励，并使用缓存"""
        p_dt = []
        use_compound = self.compound_reward and hasattr(self.distribution, 'reward_adjustment')
        
        for i in range(rel_batch.shape[0]):
            rel = rel_batch[i].item()
            dt = dt_batch[i].item()
            
            if use_compound:
                try:
                    current_entity = current_entities[i].item() if current_entities[i].dim() == 0 else current_entities[i][0].item()
                    target_entity = dst_batch[i].item() if dst_batch[i].dim() == 0 else dst_batch[i][0].item()
                    
                    # 使用缓存
                    cache_key = (rel, dt, current_entity, target_entity)
                    if cache_key in self.reward_cache:
                        p_dt.append(self.reward_cache[cache_key])
                    else:
                        adjusted_reward = self.distribution.reward_adjustment(rel, dt, current_entity, target_entity)
                        self.reward_cache[cache_key] = adjusted_reward
                        p_dt.append(adjusted_reward)
                except:
                    p_dt.append(0.0)
            else:
                # 使用缓存
                cache_key = (rel, dt)
                if cache_key in self.reward_cache:
                    p_dt.append(self.reward_cache[cache_key])
                else:
                    reward = self.distribution(rel, dt)
                    self.reward_cache[cache_key] = reward
                    p_dt.append(reward)
                
        return torch.tensor(p_dt)
