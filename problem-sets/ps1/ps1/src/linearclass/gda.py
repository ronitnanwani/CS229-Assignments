import numpy as np
import util
import math

def eval_prob(theta,x):
    p=np.inner(theta,x)
    return(1/(1+pow(math.e,-p)))

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    
    clf=GDA()
    clf.fit(x_train,y_train)

    x_eval,y_eval=util.load_dataset(valid_path,add_intercept=True)
    p_eval=clf.predict(x_eval)
    np.savetxt(save_path,p_eval)
    plot_path=save_path.replace('.txt','.png')
    util.plot(x_eval,y_eval,clf.theta,plot_path)

    acc=0.0
    n=x_eval.shape[0]
    for i in range(n):
        if((p_eval[i]>=0.5 and y_eval[i]==1) or (p_eval[i]<=0.5 and y_eval[i]==0)):
            acc+=1
    acc=(acc*100.0)/n
    print("Accuracy of GDA is: ",acc,"%")
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.theta_0=theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters

        n=x.shape[0]
        dim=x.shape[1]

        phi=0.0
        for i in range (n):
            if(y[i]==1):
                phi+=1.0
        phi=phi/n

        mu_0=np.zeros(dim)
        cnt=0
        for i in range (n):
            if(y[i]==0):
                mu_0+=x[i]
                cnt+=1
        mu_0=mu_0/cnt

        mu_1=np.zeros(dim)
        cnt=0
        for i in range(n):
            if(y[i]==1):
                mu_1+=x[i]
                cnt+=1
        mu_1=mu_1/cnt

        sigma=np.zeros((dim,dim))
        
        for i in range(n):
            if(y[i]==1):
                sigma=sigma+np.outer(x[i]-mu_1,x[i]-mu_1)
            else:
                sigma=sigma+np.outer(x[i]-mu_0,x[i]-mu_0)
        sigma=sigma/n

        sigma_inv=np.linalg.inv(sigma)
        self.theta=sigma_inv.dot(mu_1-mu_0)

        self.theta_0=(np.inner(mu_0,sigma_inv.dot(mu_0))-np.inner(mu_1,sigma_inv.dot(mu_1)))/2
        self.theta_0+=math.log(phi/(1-phi))
        self.theta=np.insert(self.theta,0,self.theta_0,axis=0)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        p_eval=[]
        n=x.shape[0]
        for i in range(n):
            p_eval.append(eval_prob(self.theta,x[i]))
        p_eval=np.array(p_eval)
        return p_eval
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
