r"""Display the bert F score of the experiment.

Example:
    python read_socre.py --exp_path exp/first_stage
                        or
    python read_socre.py --exp_path exp/second_stage
"""
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_path', type=str, default='models_normal_87')
args = parser.parse_args()

exp_path = args.exp_path

res = {}
def read_score(file_path):
    f1, precision, recall = None, None, None
    fin = open(file_path, 'r').readlines()[-15:]
    for line in fin:
        if '=== F MEASURE' in line:
            f1 = line.split()[-1]
        if '=== TOTAL TEST WORDS PRECISION' in line:
            precision = line.split()[-1]
        if '=== TOTAL TRUE WORDS RECALL' in line:
            recall = line.split()[-1]

    try:
        return {
            'f1': float(f1)*100,
            'precision': float(precision)*100,
            'recall': float(recall)*100,
        }
    except:
        return {
            'f1': .0,
            'precision': .0,
            'recall': .0,
        }


exp_dirs = os.listdir(exp_path)
for exp in exp_dirs:
    if os.path.isfile(os.path.join(exp_path, exp)):
        continue
    for file in os.listdir(os.path.join(exp_path, exp)):
        file_path = os.path.join(exp_path, exp, file)
        if file == 'train.log':
            line = open(file_path, 'r').readlines()[0]
            # seed = re.search('seed=\d+', line).group()
        if file not in ['score.txt', 'valid_score_cls.txt', 'score_cls.txt', 'score_train.txt', 'score_train_cls.txt']:
            continue
        if 'score' not in file:
            continue

        name = 'cls' if 'cls' in file else 'seg'
        res[f'{exp}-{name}'] = read_score(file_path)

## Display
def display(res):
    out_str = f'|name|f1|precision|recall|\n'
    out_str += f'|-|-|-|-|\n'
    for exp, info in res:
        f1, precision, recall = info['f1'], info['precision'], info['recall']
        # if name not in exp:
        #     continue
        pat = '(unsupervised)|(cls)|(slm)|-|_|(seg)|\d+'
        exp = re.sub(pat, '', exp)
        out_str += '|{}|{:.1f}|{:.1f}|{:.1f}|\n'.format(exp.ljust(6, " "), f1, precision, recall)
    out_str += '\n\n'

    return out_str


res = sorted(res.items(), key=lambda x: x[0])

file_str = f'\n--- {exp_path} ---\n'
file_str += display(res)

print(file_str)
with open(f'{exp_path}/read_result.txt', 'w') as f_out:
    f_out.write(file_str)

