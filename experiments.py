from td_lambda import RMSE, RandomWalk, TDLambda
import pandas as pd
import numpy as np

walker = RandomWalk()
trainset = walker.gen_trainset()

IDEAL_PREDS = [1/6, 1/3, 1/2, 2/3, 5/6]

# Experiment 1

LAMBS = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
errors = []
opt_lrs = []
for lamb in LAMBS:
    print("Lamb: {}".format(lamb))
    tmp_errs = []
    for seq_batch in trainset:
        model = TDLambda()
        model.fit_batch(seq_batch, lamb, lr=0.01)
        tmp_errs.append(RMSE(IDEAL_PREDS, model.w))
    avg_err = np.mean(tmp_errs)
    print("lambda {} avg err: {}".format(lamb, avg_err))
    errors.append(avg_err)

exp1_df = pd.DataFrame({'lambda': LAMBS,
                        'error': errors}).to_csv("exp_1_results.csv")


# Experiment 2

LAMBS = [0, 0.3, 0.8, 1]
lrs = [i/20 for i in range(13)]

errors = []

for lamb in LAMBS:
    for lr in lrs:
        model = TDLambda()
        print("Training lamba = {}, lr = {}".format(lamb, lr))
        tmp_errs = []
        for seq_batch in trainset:
            model.reset_weights()
            for seq in seq_batch:
                model._sequence(seq, lamb, lr)
            tmp_errs.append(RMSE(IDEAL_PREDS, model.w))
        errors.append((lamb, lr, np.mean(tmp_errs)))
        print("lambda {}, lr {} SCORE: {}".format(lamb, lr, np.mean(tmp_errs)))

pd.DataFrame({'lamb': [v[0] for v in errors],
              'lr': [v[1] for v in errors],
              'error': [v[2] for v in errors]}).to_csv("results/exp_2_results.csv")


# Experiment 3

LAMBS = [i/10 for i in range(11)]
lrs = [i/20 for i in range(13)]
errors = []
opt_lrs = []
for lamb in LAMBS:
    print("Lamb: {}".format(lamb))
    tmp_errs = []
    for seq_batch in trainset:
        model = TDLambda()
        model.reset_weights()
        lr_scores = []
        for lr in lrs:
            model.reset_weights()
            for seq in seq_batch:
                model._sequence(seq, lamb, lr)
            lr_scores.append(RMSE(IDEAL_PREDS, model.w))
        opt_lr, score = (lrs[np.argmin(lr_scores)], lr_scores[np.argmin(lr_scores)])
        tmp_errs.append(score)
        opt_lrs.append(opt_lr)
    avg_err = np.mean(tmp_errs)
    print("lambda {} avg err: {}".format(lamb, avg_err))
    errors.append(avg_err)

pd.DataFrame({'lambda': LAMBS,
              'error': errors}).to_csv("exp_3_results.csv")
