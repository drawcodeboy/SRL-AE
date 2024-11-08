from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def tsne_generator(data, target, model_name):
    
    model = TSNE(n_components=2)
    
    ### t-SNE ###
    
    # np.seterr(invalid='ignore') # during T-SNE if init is PCA
    embedded = model.fit_transform(data)
    # np.seterr(invalid='warn')
    
    target = target.reshape(-1, 1)
    
    data=np.concatenate((embedded, target), axis=1)
    
    labels = ['Normal', 'Anomaly']
    target = [labels[int(value)] for value in target]
    
    # print(embedded.shape, target.shape)
    df = pd.DataFrame({
        "Dim 1": embedded[:, 0],
        "Dim 2": embedded[:, 1],
        "Cluster": target
    })
    
    df_sorted = df.sort_values(by='Cluster')[::-1]
    
    plt.figure(figsize=(8, 8))
    
    # 돌려가면서 찾은 범위 값
    plt.xlim(-60, 75)
    plt.ylim(-65, 60)
    
    scatter = sns.scatterplot(data=df_sorted,
                              x="Dim 1",
                              y="Dim 2",
                              hue="Cluster",
                              legend='full',
                              palette=sns.color_palette('bright'))
    
    if model_name == 'SparLSTM-AE':
        model_name = 'Sparse LSTM-AE'
    
    scatter.set_title(f"Latent Space of {model_name} (t-SNE)")
    scatter.set_xlabel("")
    scatter.set_ylabel("")
    
    save_path = rf"figures/Latent_Space_of_{model_name}.jpg"
    
    # plt.show()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')