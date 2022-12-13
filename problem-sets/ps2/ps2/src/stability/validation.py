import util
import numpy as np

def main():
    x_a,y_a=util.load_csv('ds1_a.csv')
    x_b,y_b=util.load_csv('ds1_b.csv')
    util.plot(x_a,y_a,None,'plot_dataset_a.png')
    util.plot(x_b,y_b,None,'plot_dataset_b.png')
    

if __name__ == '__main__':
    main()