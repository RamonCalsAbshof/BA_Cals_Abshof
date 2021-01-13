import pandas as pd


def eval_restatements(data, result, restatements):
    tp, tn,fp, fn = get_formula_components(data, result, restatements)

    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1_score = calc_f1_score(precision, recall)
    accuracy = get_accuracy(tp, tn, fp, fn)

    eval_dict = {'tp': tp,
                 'tn': tn,
                 'fp': fp,
                 'fn': fn,
                 'precision': precision,
                 'recall': recall,
                 'f1_score': f1_score,
                 'accuracy': accuracy
    }
    return eval_dict


def get_formula_components(data, result, restatements):
    correct_classified = pd.merge(restatements, result, on=['ObjectID', 'Time'])

    tp = len(correct_classified)  # correct_classified
    tn = len(pd.concat([data[['ObjectID', 'Time']], result[['ObjectID', 'Time']],
                        restatements[['ObjectID', 'Time']]]).drop_duplicates(
        subset=['ObjectID', 'Time'], keep=False))  # all - result
    fp = len(pd.concat(
        [result[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
        subset=['ObjectID', 'Time'], keep=False))  # result - correct_classified
    fn = len(pd.concat(
        [restatements[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
        subset=['ObjectID', 'Time'], keep=False))  # restatements - correct_classified

    return tp, tn, fp, fn


def get_accuracy(tp, tn, fp, fn):
    if tp + fp + fn + tn > 0:
        return (tp + tn) / (tp + fp + fn + tn)
    else:
        return 0


def get_precision(tp, fp):
    if tp + fp > 0:
        return tp / (tp + fp)
    else:
        return 0


def get_recall(tp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    else:
        return 0


def get_f1_score(tp, fp, fn):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    if precision + recall > 0:
        return 2 * ((precision * recall) / (precision + recall))
    else:
        return 0


def calc_f1_score(precision, recall):
    if precision + recall > 0:
        return 2 * ((precision * recall) / (precision + recall))
    else:
        return 0