import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import copy
from genotypes import PRIMITIVES
from genotypes import Genotype
from genotypes import PARAMETERD

class MixedOp(nn.Module):

  def __init__(self, C, stride, index):
    super(MixedOp, self).__init__()
    self._C = C
    self._stride = stride
    self._ops = nn.ModuleList()
    self.primitives = copy.deepcopy(PRIMITIVES)
    self.oplist = []
    # for primitive in PRIMITIVES:# 添加混合操作
    #   op = OPS[primitive](C, stride, False)
    #   if 'pool' in primitive:
    #     op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
    #   self._ops.append(op)
    # t = PARAMETERD[index]
    op = OPS[PARAMETERD[index]](self._C, self._stride, False)
    self.primitives.remove(PARAMETERD[index])
    self._ops.append(op)
    self.oplist.append(PARAMETERD[index])

  # 添加操作
  def add_mix_op(self, index):
    if index != -1:
      op = OPS[self.primitives[index]](self._C, self._stride, False)
      if 'pool' in self.primitives[index]:
        op = nn.Sequential(op, nn.BatchNorm2d(self._C, affine=False))
      self._ops.append(op)
      self.oplist.append(self.primitives[index])
      self.primitives.pop(index)
    else:
      op = OPS['skip_connect'](self._C, self._stride, False)
      self._ops.append(op)
      self.oplist.append('skip_connect')
  def forward(self, x, weights):
    # 每一条连接输出为所有操作的加权和
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, index):
    super(Cell, self).__init__()
    self.reduction = reduction
    # 第一个输入节点
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)# 第二个输入节点
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    self.opslist = []
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1# 如果是约简单元，对于一开始的输入s0，s1到第一个中间节点的连接步长为2
        op = MixedOp(C, stride, index)#添加混合操作
        self._ops.append(op)
        self.opslist.append(op)

  # 增加操作
  def add_cell_op(self, index):
    self._ops = nn.ModuleList()
    for op in self.opslist:
      op.add_mix_op(index)
      self._ops.append(op)
    self.opslist = []
    for op in self._ops:
      self.opslist.append(op)

  # cell中的计算过程，前向传播时自动调用
  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]#当前节点的前驱节点
    offset = 0
    # 遍历每个intermediate nodes，得到每个节点的output
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))#s为当前节点i的output，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
      offset += len(states)#把当前节点i的output作为下一个节点的输入
      # states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
      states.append(s)
    # 对intermediate的output进行concat作为当前cell的输出
    # dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, index, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C #通道数
    self._num_classes = num_classes#类别数
    self._layers = layers#层数
    self._criterion = criterion#损失函数
    self._steps = steps #一个基本单元cell内有4个节点需要进行operation操作的搜索
    self._multiplier = multiplier #通道数乘数因子 因为有4个中间节点 代表通道数要扩大4倍
    self.cellslist = []
    self.num_ops = 1
    C_curr = stem_multiplier*C #当前Sequential模块的输出通道数
    # conv+bn
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    # C_prev_prev, C_prev是输入channel ;C_curr 现在是输出channel
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    #储存不同 module的list容器
    self.cells = nn.ModuleList()
    reduction_prev = False

    for i in range(layers):
      # 在 1/3 和 2/3 层减小特征size并且加倍通道，即添加约简单元
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      # 添加cell
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, index)
      reduction_prev = reduction
      self.cells += [cell]
      self.cellslist.append(cell)
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    # cell堆叠之后，后接分类
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    # 初始化alpha
    self._initialize_alphas(self.num_ops)



  # 新建network，copy alpha参数
  def new(self):
    index = 0
    model_new = Network(index, self._C, self._num_classes, self._layers, self._criterion)
    model_new.cellslist = copy.deepcopy(self.cellslist)

    model_new.num_ops = copy.deepcopy(self.num_ops)
    model_new.cells = copy.deepcopy(self.cells)
    for i in range(len(self.cellslist)):
      model_new.cellslist[i] = copy.deepcopy(self.cellslist[i])
      model_new.cells[i] = copy.deepcopy(self.cellslist[i])
    # 复制参数
    model_dict = model_new.state_dict()
    pretrained_dict = {k: v for k, v in self.state_dict().items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_new.load_state_dict(model_dict)
    # for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
    #     x.data.copy_(y.data)
    k = sum(1 for i in range(self._steps) for n in range(2 + i))  # 边数
    model_new.alphas_normal = copy.deepcopy(self.alphas_normal)
    model_new.alphas_normal.requires_grad = True
    model_new.alphas_reduce = copy.deepcopy(self.alphas_reduce)
    model_new.alphas_reduce.requires_grad = True
    model_new._arch_parameters = [
      model_new.alphas_normal,
      model_new.alphas_reduce
    ]

    return model_new.cuda()

  # 向网络中添加操作
  def add_op(self, index):
    self.num_ops += 1
    self.cells = nn.ModuleList()
    for cell in self.cellslist:
      cell.add_cell_op(index)
      self.cells.append(cell)
    self.cellslist = []
    for cell in self.cells:
      self.cellslist.append(cell)
    prev_arch_parameters = self.arch_parameters()
    # 初始化alpha
    self._initialize_alphas(self.num_ops)
    # 复制结构参数
    for x, y in zip(self.arch_parameters(), prev_arch_parameters):
      arrayx = x.cpu().detach().numpy()
      arrayy = y.cpu().detach().numpy()
      for i in range(arrayy.shape[0]):
        for j in range(arrayy.shape[1]):
          arrayx[i][j] = arrayy[i][j]
      torchx = torch.from_numpy(arrayx)
      x.data.copy_(torchx.data)
      x = x.cuda()
      x.requires_grad = True
      y = y.cuda()
      y.requires_grad = True
    print(self.cellslist[0].opslist[0].oplist)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  #初始化结构参数
  def _initialize_alphas(self, num_ops):
    k = sum(1 for i in range(self._steps) for n in range(2+i))#边数

    # self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    # self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_normal = 1e-3 * torch.randn(k, num_ops).cuda()
    self.alphas_normal.requires_grad=True
    self.alphas_reduce = 1e-3 * torch.randn(k, num_ops).cuda()
    self.alphas_reduce.requires_grad = True
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    current_primitives = self.cellslist[0].opslist[0].oplist
    print("已选操作")
    print(self.cellslist[0].opslist[0].oplist)

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        # 对于每个节点，根据其与其他节点的连接权重的最大值，来选择最优的2个连接方式（与哪两个节点之间有连接）
        # 注意这里只是选择连接的对应节点，并没有确定对应的连接op，后续确定
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                                        if 'none' not in current_primitives or k != current_primitives.index('none')))[:2]
        # 把这两条边对应的最大权重的操作找到
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if 'none' not in current_primitives or k != current_primitives.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((current_primitives[k_best], j))
        start = end
        n += 1
      return gene

    # 归一化，基于策略选取每个连接之间最优的操作
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2) #表示对节点2，3，4，5 concat
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
  def candidate_op(self):
    current_primitives = self.cellslist[0].opslist[0].oplist
    current_primitives_str = ",".join(current_primitives)
    return current_primitives_str
