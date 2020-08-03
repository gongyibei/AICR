import argparse
import torch
import math
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score



from .dataset import MidiDataset
from .model import *

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='eval or test')
    parser.add_argument("--model_path", default="./model.pth", type=str, help="the model path")
    parser.add_argument('--eval', action='store_true', help="eval")

    # get command line parameter
    args = parser.parse_args()
    model_path = args.model_path
    mode = args.eval

    # get config from config.yaml
    config = yaml_load("./config.yaml")
    base_cfg = config.get("base", {})
    model_cfg = config.get("model", {})

    init_input = model_cfg["init_input"]

    # get data path
    eval_data_path = base_cfg.get("eval_data")
    test_data_path = base_cfg.get("test_data")

    # load model
    check_point = torch.load(model_path)
    model = eval(check_point["model_name"])(*init_input, True).to(device)
    model.load_state_dict(check_point["state_dict"])
    loss_func = eval(f"F.{check_point['loss_func']}")

    #  human_scores = []
    scores = []
    labels = []

    # eval if mode is True else test
    if mode:
        logger.info(f"evaluation")
        dataset = MidiDataset(eval_data_path, base_cfg, False, False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

        model.eval()
        for i, d in enumerate(dataloader):
            x, y = d

            pred = model.forward(x.to(device))  # 500*128, 4
            x = x.contiguous().view(-1)  # 500*128
            loss = loss_func(pred.to(device), x.to(device))  # 500*128

            # Score value is larger, the more likely it is human
            scores.append(math.exp(loss.item()))
            labels.extend(list(y.numpy()))

        # cal Auc
        print(f"Auc: {roc_auc_score(labels, scores)}")
