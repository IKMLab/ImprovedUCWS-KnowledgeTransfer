r"""檢驗並得到常見中文斷詞工具的結果。
輸出結果至 `out_dir`
CWS tool: HanLP、SnowNLP 、Jieba
"""
import gc
import os
from itertools import product

import hanlp
import jieba
from snownlp import SnowNLP
import torch

# Load hanlp model.
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

def cut_lines(lines, cut_method):
    if cut_method == 'jieba':
        cut_method = lambda x: list(jieba.cut(x.strip()))
    elif cut_method == 'hanlp':
        # cut_method = lambda x: HanLP(x)['tok/fine']
        return HanLP(lines)['tok/fine']
    elif cut_method == 'snownlp':
        cut_method = lambda x: SnowNLP(x).words

    return list(map(cut_method, lines))


# a = ['天上天下唯我獨尊', '天上的雲朵很藍']
# b = cut_lines(a, 'jieba')
# print(b)

cut_method_list = ['jieba', 'snownlp', 'hanlp']
data_list = ['as', 'cityu', 'msr', 'pku', 'cityu', 'cnc', 'ctb', 'sxu', 'udc', 'wtb', 'zx']

for data, cut_method in product(data_list, cut_method_list):
    print(f"=== Current data: {data}, cut_method: {cut_method} ===")
    out_dir = f'tool_output/{cut_method}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{data}_segmented_test.txt'

    fin = open(f'data/{data}/test.txt', 'r').readlines()
    res = cut_lines(fin, cut_method)
    with open(out_path, 'w') as fout:
        for line in res:
            fout.write(f"{'  '.join(line)}\n")

    del res
    torch.cuda.empty_cache()
    gc.collect()

    SCORE_SCRIPT = 'data/score.pl'
    WORDS = f'data/{data}/words.txt'
    GOLD_FILE = f'data/{data}/test_gold.txt'
    OUTPUT = out_path
    SCORE_FILE = f'{out_dir}/{data}_segmented_test_score.txt'

    os.system(f'perl {SCORE_SCRIPT} {WORDS} {GOLD_FILE} {OUTPUT} > {SCORE_FILE}')


