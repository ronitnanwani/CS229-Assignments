import numpy as np
import util
import math

def h_theta(theta,x):
    p=np.dot(theta,x)
    return(1/(1+pow(math.e,-p)))

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path

    clf=LogisticRegression()
    clf.fit(x_train,y_train)

    x_eval,y_eval=util.load_dataset(valid_path,add_intercept=True)
    plot_path=save_path.replace('.txt','.png')
    util.plot(x_eval,y_eval,clf.theta,plot_path)

    p_eval=clf.predict(x_eval)
    np.savetxt(save_path,p_eval)

    cnt=0
    for i in range(x_eval.shape[0]):
        if((p_eval[i]>=0.5 and y_eval[i]==1) or (p_eval[i]<=0.5 and y_eval[i]==0)):
            cnt+=1
    print("Logistic Regression accuracy is: ",(100.0*cnt)/(x_eval.shape[0]),"%")
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
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
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n=x.shape[0]
        dim=x.shape[1]
        self.theta=np.zeros(dim)
        prev_theta=np.ones(dim)
        iteration=0
        while(iteration<self.max_iter and math.sqrt(np.dot(self.theta-prev_theta,self.theta-prev_theta))>self.eps):

            grad=np.zeros(dim)
            for i in range(n):
                grad=grad+(y[i]-h_theta(self.theta,x[i]))*x[i]
            
            hessian=np.zeros((dim,dim))
            for i in range(n):
                hessian=hessian-(h_theta(self.theta,x[i]))*(1-h_theta(self.theta,x[i]))*(np.outer(x[i],x[i]))
            hessian_inv=np.linalg.inv(hessian)

            prev_theta=self.theta
            self.theta=self.theta-self.step_size*hessian_inv.dot(grad)
            iteration+=1
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
        for i in range(x.shape[0]):
            p_eval.append(h_theta(self.theta,x[i]))
        p_eval=np.array(p_eval)
        return p_eval
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
