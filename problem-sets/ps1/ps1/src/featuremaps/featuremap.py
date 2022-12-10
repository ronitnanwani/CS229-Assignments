import util
import numpy as np
import matplotlib.pyplot as plt
import math

np.seterr(all='raise')


factor = 2.0

def makemap(x,k):
    arr=[]
    for i in range (k+1):
        arr.append(pow(x,i))
    arr=np.array(arr)
    return arr

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        b=np.matrix.transpose(X)
        a=np.dot(b,X)
        b=np.dot(b,y)
        self.theta=np.linalg.solve(a,b)
        
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n=X.shape[0]
        design_matrix=np.zeros((n,k+1))
        for i in range(n):
            design_matrix[i]=makemap(X[i][1],k)
        
        return design_matrix

        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n=X.shape[0]
        design_matrix=np.zeros((n,k+2))
        for i in range(n):
            design_matrix[i:i+1,0:k+1]=makemap(X[i][1],k)
            design_matrix[i][k+1]=math.sin(X[i][1])
        return design_matrix
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n=X.shape[0]
        plot_y=[]
        for i in range (n):
            plot_y.append(np.inner(self.theta,X[i]))
        plot_y=np.array(plot_y)
        return plot_y
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)
    plot_y=[]
    obj=LinearModel()
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        if sine:
            design_matrix=obj.create_sin(k,train_x)
        else:
            design_matrix=obj.create_poly(k,train_x)
            
        obj.fit(design_matrix,train_y)

        if sine:
            predict_matrix=obj.create_sin(k,plot_x)
        else:
            predict_matrix=obj.create_poly(k,plot_x)

        plot_y=obj.predict(predict_matrix)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***

    #Degree-3 Polynomial Regression
    run_exp(train_path,ks=[3],filename='degree-3_polynomial_regression.png')

    #Degree-k Polynomial Regression
    run_exp(train_path,filename='degree-k_polynomial_regression.png')

    #Other feature maps involving sin term as well
    run_exp(train_path,sine=True,filename='other_feature_maps')

    #Considering small subset of training set
    run_exp(small_path,filename='check_overfitting.png')

    #Check overfitting with other features
    run_exp(small_path,sine=True,filename='check_overfitting_with_other_features.png')

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
