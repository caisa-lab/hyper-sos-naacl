from sklearn.metrics import average_precision_score, accuracy_score, f1_score, recall_score


def acc_f1(output, labels, split, average='macro'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()

    # print(len(preds), len(labels))

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=None)[1]

    return accuracy, f1, recall
