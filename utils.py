import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def split_data(df,label):
    '''
    utilitary function used to generate data splits for the first two parts of the lab
    '''
    try:
    # Doesn't work: a value is missing
        train_data, test_data = train_test_split(df, test_size = 0.2, 
                                                 stratify=df[label])
    except:
        # Count the missing lines and drop them
        missing_rows = np.isnan(df[label])
        #print("Uh oh, {} lines missing data! Dropping them".format(np.sum(missing_rows)))
        df = df.dropna(subset=[label])
        train_data, test_data = train_test_split(df, test_size = 0.2, 
                                                 stratify=df[label])
        
    return train_data, test_data



def fit_model(X, y):
    '''
    Least squares solution of a linear model of the form y = W^Tx
    returns the estimated weights vector
    '''
    X_t = np.transpose(X) #X^T
    X_t_X = X_t.dot(X)    #X^TX
    X_T_y = X_t.dot(y)    #X^Ty
    
    #An alternative and more efficient way to compute: using a linear solver to solve the eq Ax = b
    w = linalg.solve(X_t_X, X_T_y)
    return w


def fit_logreg(X, y):
    '''
    Wraps initialization and training of Logistic regression
    '''
    logreg = LogisticRegression(C=1e20, solver='liblinear', max_iter=200) #
    logreg.fit(X, y)
    
    return logreg

def comparing_plots(xx,yy, X, y, data_1, data_2, title_1, title_2):
    '''
    utilitary function to plot results from two methods side by side. 
    It displays the training data with different colours and uses the same colours to differentiate 
    the different regions defined by the decision boundaries.
    '''
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(121)
    plt.pcolormesh(xx, yy, data_1, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title_1)
    
    plt.subplot(122)

    plt.pcolormesh(xx, yy, data_2, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title_2)
    plt.show()
    
def gaussians():
    '''
    Generates data from a multivariate Gaussian distribution.
    Means, covariances and number of samples are fixed.
    '''
    N=50
    means = np.array([[4.5, 4.5],
                      [5.5, 2.5],
                      [6.3,3.5]])
    covs = np.array([np.diag([0.5, 0.5]),
                     np.diag([0.5, 0.5]),
                     np.diag([0.5, 0.5])])
    y=[]
    points = []
    for i in range(len(means)):
        x = np.random.multivariate_normal(means[i], covs[i], N )
        points.append(x)
        y.append(i*np.ones(N)) 
    points = np.concatenate(points)
    y=np.concatenate(y)
    
    return points, y