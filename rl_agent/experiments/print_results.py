import sys
sys.path.append('..')

import os
import pandas as pd

def calc_suc_rate(data: list) -> float:
    suc_cnt = 0
    for i in data:
        suc_cnt += i['success']
    return suc_cnt / len(data)

def main(exp_name):
    # Read in the data
    data = pd.read_csv(f'./{exp_name}/eval_logs_1.txt')
    # List files in directory exp_name
    # print(os.listdir(f'./{exp_name}')

    # Calculate the success rate
    suc_rate = calc_suc_rate(data.to_dict('records'))
    print('Eval success rate = ', suc_rate)
    

if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
