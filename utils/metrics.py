import numpy as np

def base_metrics(outputs, targets):
    '''
    return TP, FP, TN, FN
    '''
    
    # batch-wise (N, ), torch.sum(tensor, dim=d)
    '''
    만약, d가 1이고, tensor가 (4, 3) shape이라면 (4, 0) + (4, 1) + (4, 2)의 텐서들이 더해진다.
    즉, 차원을 인덱스 삼아 다 더한다.
    
    #### 중요! ####
    Normal을 0으로 labeling 해두어서 0을 Positive라 보는 것이 맞다
    혹여나 헷갈리지 말자!
    '''
    TP = np.sum(np.where((outputs == targets) & (outputs == 0.), 1, 0))
    FP = np.sum(np.where((outputs != targets) & (outputs == 0.), 1, 0))
    TN = np.sum(np.where((outputs == targets) & (outputs == 1.), 1, 0))
    FN = np.sum(np.where((outputs != targets) & (outputs == 1.), 1, 0))
    
    return TP, FP, TN, FN

def get_metrics(outputs, targets, metrics):
    TP, FP, TN, FN = base_metrics(outputs, targets)
    
    metrics_dict = {}
    TP = np.where(TP == 0., 1e-6, TP)
    
    if 'Accuracy' in metrics:
        acc = (TP + TN) / (TP + TN + FP + FN); metrics_dict['Accuracy'] = acc
    if 'F1-Score' in metrics:
        f1 = 2*TP / (2*TP + FP + FN); metrics_dict['F1-Score'] = f1
    if 'Precision' in metrics:
        prec = TP / (TP + FP); metrics_dict['Precision'] = prec
    if 'Recall' in metrics:
        recall = TP / (TP + FN); metrics_dict['Recall'] = recall
    if 'Sensitivity' in metrics:
        sen = TP / (TP + FN); metrics_dict['Sensitivity'] = sen
    if 'Specificity' in metrics:
        spec = TN / (TN + FP); metrics_dict['Specificity'] = spec
    
    return metrics_dict

if __name__ == '__main__':
    a = np.array([0, 0, 1, 1, 0])
    b = np.array([0, 1, 0, 1, 0])
    
    metrics = ['Accuracy', 'Precision', 'Recall', 
               'Sensitivity', 'Specificity', 'F1-Score']
    
    m_dict = get_metrics(a, b, metrics)
    
    for key, value in m_dict.items():
        print(key, value)