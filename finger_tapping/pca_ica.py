from feature_preparation import extract_X_y
from preprocessing import simple_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import seaborn as sns
    
def plot_pca_2D(pc_x, pc_y, figsize=(8, 6)):
    """Plotting, pca index starts from 0 e.g plot pca1 and pca2 then set pc_x=0, pc_y=1"""
    plt.figure(figsize=figsize)
    sns.scatterplot(data=pca_df, x=pc_x, y=pc_y, hue='label', alpha=0.7, palette="Set1")
    plt.xlabel(f'Principal Component {pc_x + 1}')
    plt.ylabel(f'Principal Component {pc_y + 1}')
    plt.title('PCA on Finger Tapping Data')
    plt.legend(title="Class")
    plt.grid()
    plt.show()
    
def plot_ica_2D(ic_x, ic_y, figsize=(8, 6)):
    """Plotting, pca index starts from 0 e.g plot pca1 and pca2 then set pc_x=0, pc_y=1"""
    plt.figure(figsize=figsize)
    sns.scatterplot(data=ica_df, x=ic_x, y=ic_y, hue='label', alpha=0.7, palette="Set1")
    plt.xlabel(f'Independent Component {ic_x + 1}')
    plt.ylabel(f'Independent Component {ic_y + 1}')
    plt.title('ICA on Finger Tapping Data')
    plt.legend(title="Class")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def plot_ica_weights(ica_componets, figsize=(10, 4)):
    """Plot ICA component weights """
    # print(subject.info['chs'][0])
    plt.figure(figsize=figsize)
    plt.bar(range(ica_componets.mixing_.shape[0]), ica_componets.mixing_[:, 0])
    plt.title('Sensor Contributions to IC1 (Motor Laterality?)')
    plt.xlabel('Channels')
    plt.ylabel('Weight')
    plt.show()
    

if __name__ == '__main__':
    from feature_preparation import extract_X_y
    from preprocessing import simple_pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FastICA
    import pandas as pd
    
    # Load data
    subject = simple_pipeline(subject="01")
    
    # Define X,y
    X, y = extract_X_y(subject)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # Combine in dataframe
    pca_df = pd.DataFrame(X_pca)
    pca_df['label'] = y
    
    # plot pca
    plot_pca_2D(pc_x=0, pc_y=1)
    
    ##########################################
    # ICA begins here:
    ##########################################
    
    # Apply ICA
    ica = FastICA(n_components=5, max_iter=1000, tol=0.0001, random_state=42)
    X_ica = ica.fit_transform(X_scaled)

    # Prepare DataFrame
    ica_df = pd.DataFrame(X_ica)
    ica_df['label'] = y
    
    plot_ica_2D(ic_x=0, ic_y=1)
    plot_ica_weights(ica)