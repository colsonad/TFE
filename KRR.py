import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class GenerateData:
    def __init__(self,n_sample=500,n_features=20,flip_y=0.01):
        self.n_sample = n_sample
        self.n_features = n_features
        self.flip_y = flip_y

    def make_classification(self):
        """Generate some sample data"""
        x,y = make_classification(n_samples=self.n_sample, n_features=self.n_features,
                                   n_redundant=1, n_clusters_per_class=1,flip_y=flip_y,random_state=42)
        y = 2 * y - 1
        return x,y                 

    def split_data(self,X, y, test_size=0.2):
        """Split the data into training and test sets."""
        n_test = int(self.n_sample * test_size)
        indices = np.random.permutation(self.n_sample)
        X_train = X[indices[:-n_test]]
        y_train = y[indices[:-n_test]]
        X_test = X[indices[-n_test:]]
        y_test = y[indices[-n_test:]]
        return X_train, X_test, y_train, y_test  
    
from matplotlib.lines import Line2D
class KernelRidgeRegression:
    def __init__(self, lambda_param=1.0, gamma=1e-3, learning_rate=0.01, loss_target=0,initial_alpha=None,ite_max=500):
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.loss_target = loss_target
        self.initial_alpha = initial_alpha
        self.ite_max= ite_max
        self.training_loss_values= []
        self.test_loss_values= []
        self.accuracy_values = []
        self.alpha_values = np.zeros((ite_max+1,initial_alpha.shape[0]))

    def get_training_loss_values(self):
        return self.training_loss_values
    def get_test_loss_values(self):
        return self.test_loss_values
    def get_accuracy_values(self):
        return self.accuracy_values
    def get_alpha_values(self):
        return self.alpha_values
    
    def rbf_kernel(self, X1, X2):
        """Radial Basis Function kernel (Gaussian kernel)"""
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1@ X2.T
        return np.exp(-self.gamma * sq_dists)
        
    def fit_constant_stepsize(self, X,X_test, y,y_test):
        """Kernel Ridge Regression with Gradient Descent with constant stepsize"""
        n= X.shape[0]
        K = self.rbf_kernel(X,X)
        alpha = self.initial_alpha

        i=0
        self.alpha_values[i]=alpha
        while self.loss_function(X,y,alpha) > self.loss_target and i< self.ite_max: 
            self.training_loss_values.append(self.loss_function(X, y, alpha))
            self.test_loss_values.append(self.risk_function(X, X_test, y_test, alpha))
            self.accuracy_values.append(self.accuracy_score(y_test, self.predict(X, X_test, alpha)))
            
            gradient = K@((1/n * K + self.lambda_param * np.identity(n)) @ alpha - 1/n * y)
            alpha -= self.learning_rate * gradient

            i+=1
            self.alpha_values[i]=alpha

        self.training_loss_values.append(self.loss_function(X, y, alpha))
        self.test_loss_values.append(self.risk_function(X, X_test, y_test, alpha))
        self.accuracy_values.append(self.accuracy_score(y_test, self.predict(X, X_test, alpha)))
        self.ite_max = i+1

        return alpha
    
    def alpha_opt(self, X, y):
        """Optimal alpha calculation"""
        n= X.shape[0]
        K = self.rbf_kernel(X, X)
        alpha = np.linalg.inv(K+ self.lambda_param * n*np.identity(n))@y
        return alpha
    
    def predict(self,X_train, X_test, alpha):
        """Predict using the trained Kernel Ridge Regression model"""
        K_test = self.rbf_kernel(X_test, X_train)
        predictions= K_test@alpha
        return np.sign(predictions)
    
    def accuracy_score(self, y_true, y_pred):
        """Calculate the accuracy of the model"""
        return accuracy_score(y_true, y_pred)

    def loss_function(self, X, y, alpha):
        """Calculate the training loss function value = objective function"""
        K = self.rbf_kernel(X, X)
        n= X.shape[0]
        loss = 0.5/n*np.sum((y - K @ alpha) ** 2) + 0.5 * self.lambda_param * alpha.T @ K @ alpha
        return loss
      
    def risk_function(self,X_train,X_test,y_test,alpha):
        """Calculate the test loss function value"""
        K_test = self.rbf_kernel(X_test, X_train)
        n= K_test.shape[0]
        risk= 0.5/n*np.sum((y_test - K_test @ alpha) ** 2) 
        return risk
    
    def compute_errors(self, X_train, X_test, y_train, y_test, alpha):
        """Compute training and test errors"""
        train_predictions = self.predict(X_train, X_train, alpha)
        test_predictions = self.predict(X_train, X_test, alpha)
        train_error = np.mean(np.sign(train_predictions) != y_train)
        test_error = np.mean(np.sign(test_predictions) != y_test)
        return train_error, test_error
    
    def plot_level_sets(self, X_train,X_test, y_train,y_test, Model_2=None, levels=3):
        """Plot level sets of the loss and the risk function"""
        alpha_range = np.linspace(-1, 1, 60) # value of first component of alpha
        beta_range = np.linspace(-1, 1, 60) # value of second component of alpha
        A, B = np.meshgrid(alpha_range, beta_range)
        Z_l = np.zeros_like(A)
        Z_r = np.zeros_like(A)
        alpha_opt = self.alpha_opt(X_train, y_train)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                alpha_temp = alpha_opt.copy()
                alpha_temp[0] = A[i, j]
                alpha_temp[1] = B[i, j]
                Z_l[i, j] = self.loss_function(X_train, y_train, alpha_temp)
                Z_r[i, j] = self.risk_function(X_train, X_test, y_test, alpha_temp)  
    
        alpha_train = self.alpha_opt(X_train,y_train)
        min_risk_index = np.unravel_index(np.argmin(Z_r, axis=None), Z_r.shape)

        plt.figure()
        #plt.contour(A, B, Z_l, levels=[1,3,8], colors='red')
        #plt.contour(A, B, Z_r, levels=[0.5,0.6,1,1.8], colors='blue')
        plt.contour(A, B, Z_l, levels=10, colors='red',alpha=0.5)
        plt.contour(A, B, Z_r, levels=10, colors='blue',alpha=0.5,linestyles='dotted')

        plt.plot(Model_2.alpha_values[:Model_2.ite_max,0],Model_2.alpha_values[:Model_2.ite_max,1],color='green')
        plt.plot(self.alpha_values[:self.ite_max,0],self.alpha_values[:self.ite_max,1],color='black')
        # for alpha in Model_2.alpha_values[:Model_2.ite_max]:
        #     plt.scatter(alpha[0],alpha[1],color='green')  
        # for alpha in self.alpha_values[:self.ite_max]:
        #     plt.scatter(alpha[0],alpha[1],color='black') 
        plt.scatter(alpha_train[0],alpha_train[1],color='red')
        plt.scatter(A[min_risk_index], B[min_risk_index], color='blue')

        plt.xlabel('Alpha 1')
        plt.ylabel('Alpha 2')
        plt.title('Level Sets of the Loss Function')
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='Loss Function'),
                   Line2D([0], [0], color='blue', lw=2, label='Test loss Function'),
                   Line2D([0],[0],color='green', marker='o',linestyle='None',label='stepsize = 1.9/L'),
                   Line2D([0],[0],color='black', marker='o',linestyle='None',label='stepsize = 1/L')]
        
        plt.legend(handles=legend_elements)
        plt.grid(True)
        #plt.savefig("Level_set.pdf")
        plt.show()

class Operators:
    def __init__(self,K,lambda_param):
        self.n= K.shape[0]
        self.lambda_param=lambda_param 
        self.K = K 
        self.cond_T= ((1/n * K.T@K + self.lambda_param * K))

        n= K.shape[0]
        eigenvals,_ = np.linalg.eigh(1/n * K.T@K + self.lambda_param * K)
        L = np.max(eigenvals)
        self.L=L
         
    def condition_number_T(self): 
        return self.cond_T 
    def largest_eigenvalue(self):
        return np.max(np.linalg.eigvals(self.cond_T))
    def smallest_eigenvalue(self):
        return np.min(np.linalg.eigvals(self.cond_T))
    def Lipschitz_constant(self):
        return self.L

n_samples=500
n_features=30
flip_y=0.1

Data= GenerateData(n_samples,n_features,flip_y)
x,y = Data.make_classification()
np.random.seed(42)  # Set the random seed for reproducibility
X_train, X_test, y_train, y_test = Data.split_data(x, y, test_size=0.2)

lambda_param=0
gamma=0.1
loss_target = 0.5

Model = KernelRidgeRegression(lambda_param=lambda_param, gamma=gamma,initial_alpha=np.zeros(X_train.shape[0])) 
Op = Operators(Model.rbf_kernel(X_train,X_train),lambda_param)
L = Op.Lipschitz_constant()
mu = Op.smallest_eigenvalue()
print(1.9/L)
print(1/L)

l_1 = 1.9/L
l_2 = 1/L

initial_alpha = initial_alpha = np.random.uniform(-1, 1,X_train.shape[0])

Model_1 = KernelRidgeRegression(lambda_param=lambda_param, gamma=gamma, learning_rate=l_1, loss_target=loss_target,initial_alpha=initial_alpha.copy())
alpha_1 = Model_1.fit_constant_stepsize(X_train,X_test, y_train,y_test)
predictions_1 = Model_1.predict(X_train, X_test, alpha_1)
accuracy_1 = Model_1.accuracy_score(y_test, predictions_1)
loss_values_1 = Model_1.get_training_loss_values()
test_values_1 = Model_1.get_test_loss_values()
accuracy_values_1 = Model_1.get_accuracy_values()
alpha_1_values = Model_1.get_alpha_values()


alpha_opt = Model_1.alpha_opt(X_train, y_train)

Model_2 = KernelRidgeRegression(lambda_param=lambda_param, gamma=gamma, learning_rate=l_2, loss_target=loss_target,initial_alpha=initial_alpha.copy())
alpha_2 = Model_2.fit_constant_stepsize(X_train,X_test, y_train,y_test)
predictions_2 = Model_2.predict(X_train, X_test, alpha_2)
accuracy_2 = Model_2.accuracy_score(y_test, predictions_2)
loss_values_2 = Model_2.get_training_loss_values()
test_values_2 = Model_2.get_test_loss_values()
accuracy_values_2 = Model_2.get_accuracy_values()
alpha_2_values = Model_2.get_alpha_values()

Model_1.plot_level_sets(X_train,X_test, y_train,y_test,Model_2=Model_2)


plt.figure()
plt.plot(range(len(loss_values_2)), loss_values_2, label=f'stepsize = {l_2}')
plt.plot(range(len(loss_values_1)), loss_values_1, label=f'stepsize = {l_1}')
plt.axhline(y=Model_1.loss_function(X_train, y_train, alpha_opt), color='r', linestyle='-', label='Optimal alpha 1')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.legend()
plt.grid(True)
plt.show()

"""
plt.figure()
plt.plot(range(len(accuracy_values_1)), accuracy_values_1, label=f'Lambda = {l_1}')
plt.plot(range(len(accuracy_values_2)), accuracy_values_2, label=f'Lambda = {l_2}')
plt.axhline(y=Model_1.accuracy_score(y_test, Model_1.predict(X_train, X_test, alpha_opt)), color='r', linestyle='-', label='Optimal alpha 1')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iterations')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(loss_values_1, accuracy_values_1, label=f'Lambda = {l_1}')
plt.plot(loss_values_2, accuracy_values_2, label=f'Lambda = {l_2}')
plt.plot(Model_1.loss_function(X_train,y_train,alpha_opt),Model_1.accuracy_score(y_test,Model_1.predict(X_train, X_test, alpha_opt)),'ro',label='Optimal alpha 1')
plt.xlabel('Loss')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(len(test_values_1)), test_values_1, label=f'Lambda = {l_1}')
plt.plot(range(len(test_values_2)), test_values_2, label=f'Lambda = {l_2}')
plt.axhline(y=Model_1.risk_function(X_train,X_test,y_test,alpha_opt), color='r', linestyle='-', label='Optimal alpha 1')
plt.xlabel('Iterations')
plt.ylabel('Risk')
plt.title('Risk vs Iterations')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(loss_values_1, test_values_1, label=f'Lambda = {l_1}')
plt.plot(loss_values_2, test_values_2, label=f'Lambda = {l_2}')
plt.plot(Model_1.loss_function(X_train,y_train,alpha_opt),Model_1.risk_function(X_train,X_test,y_test,alpha_opt),'ro',label='Optimal alpha 1')
plt.xlabel('Loss')
plt.ylabel('Risk')
plt.title('Risk vs Loss')
plt.legend()
plt.grid(True)
plt.show()

"""
learning_rates_s = np.linspace(1/L, 1.9/L, num=15)

accuracies = []
test_loss = []
training_loss =[]
traini_accuracy=[]
loss_target = 0.8

initial_alpha = np.random.uniform(-1, 1,X_train.shape[0])

for lr in learning_rates_s:
    Model = KernelRidgeRegression(lambda_param=lambda_param, gamma=gamma, learning_rate=lr, loss_target=loss_target,initial_alpha=initial_alpha.copy())
    alpha = Model.fit_constant_stepsize(X_train, X_test, y_train, y_test)
    predictions = Model.predict(X_train, X_test, alpha)
    accuracy = Model.accuracy_score(y_test, predictions)
    loss = Model.risk_function(X_train, X_test, y_test, alpha)
    train_loss = Model.loss_function(X_train,y_train,alpha)
    train_accuracy = Model.accuracy_score(y_train, Model.predict(X_train, X_train, alpha))
    accuracies.append(accuracy)
    test_loss.append(loss)
    training_loss.append(train_loss)
    traini_accuracy.append(train_accuracy)
    print(Model.ite_max)

plt.figure()
plt.plot(learning_rates_s, training_loss, marker='o', linestyle='-', label='Loss')
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Loss vs Learning Rate')
plt.grid(True)
#plt.savefig('Train_loss.pdf')
plt.show()

plt.figure()
plt.plot(learning_rates_s, accuracies, marker='o', linestyle='-', label='Test Accuracy')
plt.plot(learning_rates_s, traini_accuracy, marker='o', linestyle='-', label='Train Accuracy')
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
plt.grid(True)
#plt.savefig('Accuracy.pdf')
plt.show()

plt.figure()
plt.plot(learning_rates_s, test_loss, marker='o', linestyle='-')
plt.legend()
plt.xlabel('Stepsize')
plt.ylabel('Test Loss')
plt.title('Test loss function')
plt.grid(True)
#plt.savefig('Test_loss.pdf')
plt.show()
