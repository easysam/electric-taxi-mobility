import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from utils import display
from sklearn.metrics import ndcg_score


def ndcg_at_k(_true_score, k=10):
    _len = len(_true_score)
    _true_score = _true_score < 10
    _score = [0] * _len
    _score[:10] = [1] * 10
    return ndcg_score([_true_score], [_score], k=k)


if __name__ == '__main__':
    display.configure_pandas()
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    generated_14et = pd.read_parquet('result/generation/result_transition_14et')
    generated_14all = pd.read_parquet('result/generation/result_transition_14all')
    generated_pred = pd.read_parquet('result/generation/result_transition_pred')
    true_score_14et = generated_14et['station'].value_counts().index
    true_score_14all = generated_14all['station'].value_counts().index
    true_score_pred = generated_pred['station'].value_counts().index

    for _k in range(10, 151, 10):
        print('14et/HR@{}: {:.4f}'.format(_k, (true_score_14et[:_k] < _k).sum() / _k))
        print('14et/NDCG@{}: {:.4f}'.format(_k, ndcg_at_k(true_score_14et, k=_k)))
        print('14all/HR@{}: {:.4f}'.format(_k, (true_score_14all[:_k] < _k).sum() / _k))
        print('14all/NDCG@{}: {:.4f}'.format(_k, ndcg_at_k(true_score_14all, k=_k)))
        print('pred/HR@{}: {:.4f}'.format(_k, (true_score_pred[:_k] < _k).sum() / _k))
        print('pred/NDCG@{}: {:.4f}'.format(_k, ndcg_at_k(true_score_pred, k=_k)))
