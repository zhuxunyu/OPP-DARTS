import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# 分配单gpu
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
# 分配多gpu
# parser.add_argument('--gpu', type=str, default='cuda:0,1,2', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--fine_tune', type=int, default=0, help='微调次数')
parser.add_argument('--temperature', type=int, default=4, help='温度系数')
parser.add_argument('--temperature_step', type=int, default=0.6, help='退火速率')
parser.add_argument('--add_op_epoch', type=int, default=2, help='增加op的epoch')
parser.add_argument('--init_op', type=int, default=4, help='初始op次数')
parser.add_argument('--candidate_op', type=int, default=6, help='候选op次数')
parser.add_argument('--fine_tune_learning_rate', type=float, default=0.0005, help='微调学习率')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  # 用于指定随机数生成时所用算法开始的整数值。
  np.random.seed(args.seed)
  # 分配gpu
  torch.cuda.set_device(args.gpu)
  # 实现网络加速
  cudnn.benchmark = True
  # 设置随机初始化的种子，编号固定，每次获取的随机数固定。
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  # logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  # 定义交叉熵损失函数
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  # 数据处理
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)#50000
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))#25000

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  candidate_network = []
  probability_network = []
  logging.info('初始化模型')
  for i in range(args.init_op):
    # 初始化选择模型
    candidate_model = Network(i, args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    candidate_model = candidate_model.cuda()
    candidate_network.append(candidate_model)
    loss = 0
    batch_number = 0
    for step, (input, target) in enumerate(train_queue):
      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda(async=True)
      with torch.no_grad():
        logits = candidate_model(input)
      loss += criterion(logits, target)
      batch_number = step
      # print(loss)
    probability_network.append(loss / (batch_number + 1))
    print(loss / (batch_number + 1))
  # probability_network = [1 / (loss * args.temperature) for loss in probability_network]
  # probability_network = [loss / args.temperature for loss in probability_network]
  # probability_network = torch.tensor(probability_network, dtype=torch.double)
  # probability_network = F.softmax(probability_network, dim=-1)
  # probability_network = probability_network.numpy().tolist()
  logging.info("候选操作的loss：")
  logging.info(probability_network)
  max_probability_network = 0
  index_max_op = 0
  for i in range(len(probability_network)):
    if max_probability_network < probability_network[i]:
      max_probability_network = probability_network[i]
      index_max_op = i
  model = candidate_network[index_max_op]
  # model = np.random.choice(candidate_network, p=probability_network)
  # 网络权重优化器
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  # 余弦退火调整学习率
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  #logging.info("param size = %fMB", utils.count_parameters_in_MB(model))  # 计算模型的计算量
  candidate_op = args.candidate_op
  temperature = args.temperature * args.temperature_step
  optimizer2 = torch.optim.SGD(
    model.parameters(),
    args.fine_tune_learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
  # # 微调
  # logging.info('微调')
  # for i in range(args.fine_tune):
  #   for step, (input, target) in enumerate(train_queue):
  #     model.train()
  #     input = Variable(input, requires_grad=False).cuda()
  #     target = Variable(target, requires_grad=False).cuda(async=True)
  #     optimizer2.zero_grad()
  #     logits = model(input)
  #     loss = criterion(logits, target)  # 预测值logits和真实值target的loss
  #     if step % args.report_freq == 0:
  #       logging.info('微调 %03d %e', step, loss.data)
  #     loss.backward()  # 反向传播，计算梯度
  #     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
  #     optimizer2.step()


  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    if epoch != 0 and epoch % args.add_op_epoch == 0 and candidate_op > 0:
    # if epoch == 1 or  epoch == 2:
    # if epoch == -1:
      logging.info('选择操作')

      candidate_network = []
      probability_network = []
      model = model.cpu()
      for op in range(candidate_op):

        # 添加操作
        candidate_model = model.new()
        candidate_model.add_op(op)
        candidate_model = candidate_model.cuda()
        candidate_network.append(candidate_model)
        loss = 0
        batch_number = 0
        for step, (input, target) in enumerate(train_queue):
          # candidate_model.train()
          input = Variable(input, requires_grad=False).cuda()
          target = Variable(target, requires_grad=False).cuda(async=True)
          with torch.no_grad():
            logits = candidate_model(input)
          loss += criterion(logits, target)
          batch_number = step
        probability_network.append(loss / (batch_number + 1))

      candidate_op -= 1
      # probability_network = [1 / (loss * temperature) for loss in probability_network]
      # probability_network = [loss / args.temperature for loss in probability_network]
      # probability_network = torch.tensor(probability_network, dtype=torch.double)
      # probability_network = F.softmax(probability_network, dim=-1)
      # probability_network = probability_network.numpy().tolist()
      # #probability_network_str = ",".join(probability_network)
      # logging.info("候选操作的概率：")
      # logging.info(probability_network)
      # model = np.random.choice(candidate_network, p=probability_network)
      logging.info("候选操作的loss：")
      logging.info(probability_network)
      max_probability_network = 0
      index_max_op = 0
      for i in range(len(probability_network)):
        if max_probability_network < probability_network[i]:
          max_probability_network = probability_network[i]
          index_max_op = i
      model = candidate_network[index_max_op]
      # 网络权重优化器
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
      # 余弦退火调整学习率
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs - epoch), eta_min=args.learning_rate_min)
      temperature = temperature * args.temperature_step
      architect = Architect(model, args)
      optimizer2 = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
      # 微调
      logging.info('微调')
      for i in range(args.fine_tune):
        for step, (input, target) in enumerate(train_queue):
          model.train()
          input = Variable(input, requires_grad=False).cuda()
          target = Variable(target, requires_grad=False).cuda(async=True)
          optimizer2.zero_grad()
          logits = model(input)
          loss = criterion(logits, target)  # 预测值logits和真实值target的loss
          if step % args.report_freq == 0:
            logging.info('微调 %03d %e', step, loss.data)
          loss.backward()  # 反向传播，计算梯度
          nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
          optimizer2.step()

    if candidate_op == 0 and epoch % args.add_op_epoch == 0 and epoch > (args.add_op_epoch * args.candidate_op):
    # if epoch == 1:
      logging.info('添加skip_connect操作')
      candidate_op -= 1
      model = model.cpu()
      model.add_op(-1)
      model = model.cuda()
      # 网络权重优化器
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
      # 余弦退火调整学习率
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs - epoch), eta_min=args.learning_rate_min)
      architect = Architect(model, args)
      optimizer2 = torch.optim.SGD(
        model.parameters(),
        args.fine_tune_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
      # 微调
      logging.info('微调')
      for i in range(args.fine_tune):
        for step, (input, target) in enumerate(train_queue):
          model.train()
          input = Variable(input, requires_grad=False).cuda()
          target = Variable(target, requires_grad=False).cuda(async=True)
          optimizer2.zero_grad()
          logits = model(input)
          loss = criterion(logits, target)  # 预测值logits和真实值target的loss
          if step % args.report_freq == 0:
            logging.info('微调 %03d %e', step, loss.data)
          loss.backward()  # 反向传播，计算梯度
          nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
          optimizer2.step()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))  # 计算模型的计算量
    logging.info("已选的op集合" + model.candidate_op())
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    scheduler.step()
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()# 用于保存loss的值
  top1 = utils.AvgrageMeter()# 前1预测正确的概率
  top5 = utils.AvgrageMeter()# 前5预测正确的概率
  # 每个step取出一个batch，batchsize是64（256个数据对）
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    #更新α是用validation set进行更新的，所以我们每次都从valid_queue拿出一个batch传入architect.step()
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)
    # (async=True)
    # 更新α，unrolled是true就是用论文的公式进行α的更新
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)#预测值logits和真实值target的loss

    loss.backward()#反向传播，计算梯度
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)#梯度裁剪
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)
      logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

