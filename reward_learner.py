import argparse
import os
import pickle
import time  # 添加时间模块用于性能测量

import numpy as np
import torch  # 添加torch导入

from dataset.baseDataset import baseDataset

from model.TemporalAttention import TemporalRewardLearner

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Temporal Reward Learning')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str)
    parser.add_argument('--outfile', default='attention_rewards.pkl', type=str)
    parser.add_argument('--hidden_dim', default=304, type=int)
    parser.add_argument('--time_span', default=24, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)  # 添加权重衰减参数
    parser.add_argument('--label_smoothing', default=0.1, type=float)  # 添加标签平滑参数
    parser.add_argument('--device', default='cuda', type=str, help='计算设备 (cuda/cpu)')  # 添加设备参数
    parser.add_argument('--num_workers', default=2, type=int, help='数据加载线程数')  # 添加数据加载线程数参数
    args = parser.parse_args()

    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行计算")
        args.device = 'cpu'
    else:
        # 打印GPU信息
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

    # 构建数据文件路径
    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None

    # 创建数据集对象
    dataset = baseDataset(trainF, testF, statF, validF)

    # 数据统计
    print(f"Number of entities: {dataset.num_e}")
    print(f"Number of relations: {dataset.num_r}")
    print(f"Number of training quadruples: {len(dataset.trainQuadruples)}")
    
    # 训练改进模型
    print("Initializing reward learner...")
    start_time = time.time()  # 记录开始时间
    
    # 创建学习器
    reward_learner = TemporalRewardLearner(
        quadruples=dataset.trainQuadruples,
        num_entities=dataset.num_e,
        num_relations=dataset.num_r,
        hidden_dim=args.hidden_dim,
        time_span=args.time_span,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,  # 传递设备参数
        label_smoothing=args.label_smoothing  # 传递标签平滑参数
    )

    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\n训练完成，耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")

    # 保存结果
    rewards = reward_learner.rewards
    # 结果分析
    print("\nReward Distribution Analysis:")
    print(f"- Mean ± Std: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"- Median: {np.median(rewards):.4f}")
    print(f"- 25-75 Percentile: {np.percentile(rewards, 25):.4f} - {np.percentile(rewards, 75):.4f}")

    # 保存奖励文件
    out_path = os.path.join(args.data_dir, args.outfile)
    pickle.dump(rewards, open(out_path, 'wb'))
    print(f"Rewards saved to {out_path}")


if __name__ == '__main__':
    main()