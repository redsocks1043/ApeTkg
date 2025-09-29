import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.TemporalAttention import  AttentionRewardDistribution, TemporalRewardLearner, DynamicRewardAdjustment, CompoundRewardDistribution
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
import os
import pickle

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main2.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not.')
    parser.add_argument('--data_path', type=str, default='data/ICEWS14', help='Path to data.')
    parser.add_argument('--do_train', action='store_true', help='whether to train.')
    parser.add_argument('--do_test', action='store_true', help='whether to test.')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=400, type=int, help='max training epochs.')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=400, type=int, help='validation frequency.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')
    parser.add_argument('--save_epoch', default=30, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')

    # Test Params
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=100, type=int, help='the beam number of the beam search.')
    parser.add_argument('--test_inductive', action='store_true', help='whether to verify inductive inference performance.')
    parser.add_argument('--IM', action='store_true', help='whether to use IM module.')
    parser.add_argument('--mu', default=0.1, type=float, help='the hyperparameter of IM module.')

    # Agent Params
    parser.add_argument('--ent_dim', default=100, type=int, help='Embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='Embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the LSTM hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the MLP hidden layer')
    parser.add_argument('--time_dim', default=20, type=int, help='Embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')

    # Episode Params
    parser.add_argument('--path_length', default=3, type=int, help='the agent search path length.')
    parser.add_argument('--max_action_num', default=50, type=int, help='the max candidate actions number.')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.0, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')

    # reward shaping params
    parser.add_argument('--reward_shaping', action='store_true', help='whether to use reward shaping.')
    parser.add_argument('--time_span', default=24, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    # parser.add_argument('--alphas_pkl', default='dirchlet_alphas.pkl', type=str,
    #                     help='the file storing the alpha parameters of the Dirichlet distribution.')
    # parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')

    # 添加注意力模型参数
    parser.add_argument('--attention_rewards_file', default='attention_rewards.pkl', type=str,
                        help='Path to save/load attention rewards')
    parser.add_argument('--attention_hidden_dim', default=304, type=int,
                        help='Hidden dimension for attention model')
    parser.add_argument('--attention_epochs', default=50, type=int,
                        help='Number of epochs for training attention model')
    
    # 添加复合奖励参数
    parser.add_argument('--compound_reward', action='store_true', help='whether to use compound reward (R+ΔR).')
    parser.add_argument('--reward_delta_dim', default=100, type=int, help='Hidden dimension for reward delta model')
    parser.add_argument('--max_delta_scale', default=0.5, type=float, help='Maximum scale factor for reward delta')
    
    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel):
    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
        # 'num_heads': args.num_heads,  # 新增：注意力头数
        # 'history_len': args.history_len,  # 新增：历史序列长度
    }
    return config

def main(args):
    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'train.txt')
    test_path = os.path.join(args.data_path, 'test.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'valid.txt')

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset  = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        logging.info(f"状态动作空间文件 {state_actions_path} 不存在，将使用 None")
        state_action_space = None
    else:
        logging.info(f"尝试加载状态动作空间文件: {state_actions_path}")
        try:
            with open(state_actions_path, 'rb') as f:
                state_action_space = pickle.load(f)
            logging.info(f"成功加载状态动作空间文件: {state_actions_path}")
        except ModuleNotFoundError as e:
            logging.warning(f"由于 NumPy 兼容性问题，无法加载 {state_actions_path}: {str(e)}，使用 None")
            state_action_space = None
        except Exception as e:
            logging.error(f"加载 {state_actions_path} 失败: {str(e)}，使用 None")
            state_action_space = None
    env = Env(baseData.allQuadruples, config, state_action_space)

    # Create episode controller
    episode = Episode(env, agent, config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        # 只在训练模式下加载优化器状态
        if args.do_train and not args.do_test:
            try:
                optimizer.load_state_dict(params['optimizer_state_dict'])
                logging.info('成功加载优化器状态')
            except ValueError as e:
                logging.warning(f"无法加载优化器状态，错误信息：{str(e)}")
                logging.info("继续执行而不加载优化器状态...")
        else:
            logging.info('测试模式：跳过加载优化器状态')

        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    ######################Training and Testing###########################
    if args.reward_shaping:
        reward_file = os.path.join(args.data_path, args.attention_rewards_file)

        if os.path.exists(reward_file):
            # 加载已计算的注意力奖励
            logging.info('Loading pre-computed attention rewards...')
            attention_rewards = pickle.load(open(reward_file, 'rb'))
        else:
            # 创建并训练注意力奖励学习器
            logging.info('Training attention reward model...')
            reward_learner = TemporalRewardLearner(
                quadruples=baseData.trainQuadruples,
                num_entities=baseData.num_e,
                num_relations=baseData.num_r,
                hidden_dim=args.attention_hidden_dim,
                time_span=args.time_span,
                device='cuda' if args.cuda else 'cpu',
                lr=args.lr,
                batch_size=args.batch_size,
                num_epochs=args.attention_epochs
            )

            # 获取计算的奖励
            attention_rewards = reward_learner.rewards

            # 保存计算的奖励
            pickle.dump(attention_rewards, open(reward_file, 'wb'))
            logging.info(f'Saved attention rewards to {reward_file}')

            # 输出奖励统计信息
            logging.info(f"Reward statistics:")
            logging.info(f"Mean: {np.mean(attention_rewards):.4f}")
            logging.info(f"Std: {np.std(attention_rewards):.4f}")
            logging.info(f"Min: {np.min(attention_rewards):.4f}")
            logging.info(f"Max: {np.max(attention_rewards):.4f}")

        # 创建注意力奖励分布
        base_distributions = AttentionRewardDistribution(
            attention_rewards,
            'cuda' if args.cuda else 'cpu'
        )
        
        # 如果启用复合奖励，创建动态奖励调整模块
        if args.compound_reward:
            logging.info('Initializing compound reward mechanism (R+ΔR)...')
            reward_adjustment = DynamicRewardAdjustment(
                num_entities=baseData.num_e,
                num_relations=baseData.num_r,
                hidden_dim=args.reward_delta_dim,
                device='cuda' if args.cuda else 'cpu'
            )
            if args.cuda:
                reward_adjustment = reward_adjustment.cuda()
                
            # 创建复合奖励分布
            distributions = CompoundRewardDistribution(
                base_distributions,
                reward_adjustment,
                'cuda' if args.cuda else 'cpu'
            )
            
            # 将奖励调整模块添加到优化器中
            optimizer = torch.optim.Adam(
                list(episode.parameters()) + list(reward_adjustment.parameters()),
                lr=args.lr, 
                weight_decay=0.00001
            )
        else:
            distributions = base_distributions
    else:
        distributions = None
    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence, distributions)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            # 如果使用复合奖励，更新缩放因子
            if args.reward_shaping and args.compound_reward:
                distributions.update_scale_factor(i, args.max_epochs)
                
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(args.save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')
        metrics = tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'])
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)

