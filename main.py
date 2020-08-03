import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from model import *
from midi import *
from dataset import *
import sys
from sklearn.metrics import roc_auc_score
from datetime import datetime
import yaml
from log import LOG

def yaml_load(config_path: str) -> dict:
    with open(config_path) as f:
        param = yaml.safe_load(f)
    return param

def train_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    for i, (batch_x, batch_y) in enumerate(data_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        # if (i + 1) % 10 == 0:
        #     accuracy = (num_correct / num_total) * 100
        #     sys.stdout.write(f'\r准确率：{accuracy:.2f}')
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    # sys.stdout.write('\r\r')
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    for batch_x, batch_y in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def evaluate_auc(data_loader, model, device):
    labels = []
    scores = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        # batch_y = batch_y.to(device)
        batch_out = model(batch_x)
        scores.extend(list(batch_out.cpu().detach().numpy()[:, 1]))
        labels.extend(list(batch_y.numpy()))
    return roc_auc_score(labels, scores)

def train(need_eval=True):
    LOG.info('====================================================================')
    LOG.info('=                          Start Training                          =')
    LOG.info('====================================================================')

   # init device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load config
    config = yaml_load('./config.yaml')
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})
    eval_cfg = config.get('eval', {})

    # load model config
    modelname = model_cfg.get('model', 'resnet')
    criterion = model_cfg.get('criterion', 'CrossEntropyLoss')
    optimizer = model_cfg.get('optimizer', 'Adam')

    # load train config
    epochs = train_cfg.get('epochs', 100)
    train_batch_size = train_cfg.get('batch_size', 100)
    trainlist = train_cfg.get('trainlist', [])
    lr = train_cfg.get('lr', 0.0001)
    vs = train_cfg.get('vs', 0.5)
    LOG.info(f'epochs:{epochs}, training batch size:{train_batch_size}, learning rate:{lr}, split ration:{vs}')

    # load eval config
    if need_eval:
        eval_batch_size = eval_cfg.get('batch_size', 20)
        evallist = eval_cfg.get('evallist', [])

    # load training dataset
    LOG.info('Start load training dataset...')
    dataset = MidiDataset(trainlist)
    train_size = int(len(dataset)*vs)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # load evaluation dataset
    if need_eval:
        LOG.info('Start load evaluation dataset...')
        eval_dataset = MidiDataset(evallist)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True, drop_last=True)

    # create model
    model = eval(modelname)().to(device)
    criterion = eval(f'nn.{criterion}')()
    optimizer = eval(f'optim.{optimizer}')(model.parameters(), lr=lr)
    LOG.info(f'Model created! model name:{modelname} \n model:{model}, \ncriterion:{criterion}, \noptimizer:{optimizer}')
    model_dir = os.path.join(os.path.abspath('.'), 'models', modelname, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    LOG.info(f'Set model dir to {model_dir}')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(epochs):
        LOG.info(f'===========================epoch:{epoch}===========================')

        model.train()
        running_loss, train_accuracy = train_epoch(train_loader, model, criterion, optimizer, device)
        train_info = f'训练集> Acc：{train_accuracy:7.4f}%, Loss:{running_loss:.6f}'
        LOG.info(train_info)

        model.eval()
        test_accuracy = evaluate_accuracy(test_loader, model, device)
        test_auc = evaluate_auc(test_loader, model, device)
        test_info = f'测试集> Acc：{test_accuracy:7.4f}%, AUC:{test_auc:.6f}'
        LOG.info(test_info)

        if need_eval:
            eval_accuracy = evaluate_accuracy(eval_loader, model, device)
            eval_auc = evaluate_auc(eval_loader, model, device)
            eval_info = f'验证集> Acc：{eval_accuracy:7.4f}%, AUC:{eval_auc:.6f}'
            LOG.info(eval_info)

        model_path = os.path.join(model_dir, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        LOG.info(f'Save model to "{model_path}"')


def evaluate():
    LOG.info('====================================================================')
    LOG.info('=                         Start Evaluation                         =')
    LOG.info('====================================================================')

    # init device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load config
    config = yaml_load("./config.yaml")
    model_cfg = config.get('model', {})
    eval_cfg = config.get("eval", {})

    # load model config
    modelname = model_cfg.get('model', 'resnet')

    # eval_cfg
    modelpath = eval_cfg.get('modelpath', '')
    batch_size = eval_cfg.get('batch_size', 1)
    dirlist = eval_cfg.get('dirlist', [])

    # load dataset
    dataset = MidiDataset(dirlist)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # load model
    model = eval(modelname)()
    model.load_state_dict(torch.load(modelpath))
    model.to(device)
    LOG.info('Model loaded! mldel path:"{modelpath}"')

    # evaluate
    model.eval()
    accuracy = evaluate_accuracy(eval_loader, model, device)
    auc = evaluate_auc(eval_loader, model, device)
    eval_info = f'测试集：{accuracy:.4f}, AUC:{auc:.4f}'
    LOG.info(eval_info)

if __name__ == '__main__':
    train()
    # evaluate()

