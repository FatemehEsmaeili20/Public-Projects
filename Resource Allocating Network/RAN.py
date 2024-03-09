import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class ResourceAllocatingNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.centers = []
        self.widths = []
        self.weights = []
        self.gamma = np.zeros(output_size)

    def gradient_descent(self, input_vector, x, error, alpha):

        # updating the value of gamma according to this formula: gamma = gamma + alpha * error
        self.gamma  += alpha * error

        # updating the values of weights and centers
        for j in range(len(self.centers)):
            # Because the index of list starts from zero but j starts from 1, the indexes of variables are j-1
            # updating weight(h) according to : h = h + alpha * error * x
            xj = x[j-1]
            er_mul_xj = np.zeros(len(error))
            for i in range(len(error)):
                er_mul_xj[i-1] = alpha * xj * error[i-1] # alpha and xj are scalr but error has a dimension of output_size
            weight = self.weights[j-1] # weight and weight1 are a temporary variable to calculate weights and have dimension of output_size
            self.weights[j-1] = weight + er_mul_xj
            # Updating center according to : c = c + (2 * alpha / width) * (Input - c) * x * (error * h)
            coef1 = 2 * (alpha / self.widths[j-1]) * xj
            # Calculating "vector multiplication" of "error and weight(h)"
            er_mul_weight = 0
            for i in range(len(error)):
                er_mul_weight += error[i - 1] * weight[i - 1]
            coef2 = coef1 * er_mul_weight
            center = self.centers[j-1]  # center is a temporary variable to calculate centers and has a dimension of input_size
            # Calculating "Input - c"
            coef3 = np.zeros(len(center))
            for i in range(len(center)):
                coef3[i-1] = input_vector[i-1] - center[i-1]
            # Updating center
            for i in range(len(center)):
                coef3[i-1] = coef3[i-1] * coef2
            center += coef3
            self.centers[j-1] = center

            # Gradient descent for updating widths
            width = self.widths[j-1]  # width is a temporary variable to calculate widths and has a dimension of centers
            coef4 = np.linalg.norm(coef3) / ((self.widths[j-1]) ** 2)
            width += coef4
            self.widths[j-1] = width

        #print('Centers in this Gradient Descent are changed as follows: ')
        #print(self.centers)
        #print('Weights in this Gradient Descent are changed as follows: ')
        #print(self.weights)
        #print('widths in this Gradient Descent are changed as follows: ')
        #print(self.widths)
        #print('gamma in this Gradient Descent are changed as follows: ')
        ##print(self.gamma)

    def train(self, input_vectors, target_outputs, epsilon, delta_max, delta_min, k, tau, alpha, num_epochs, error_threshold):
        # The distance "delta" is the scale of resolution that the network is fitting at the ith input presentation.
        # The learning starts with delta = delta_max
        delta = delta_max
        # gamma is the default output of the network when none of the first-layer units are active.
        # At the first step of training, the value of gamma is the first output(Y).
        self.gamma = target_outputs[0]
        # At the first step of training, the value of the first center is the first input data(X)
        self.centers.append(input_vectors[0])
        width = k * delta # The value of the width of the first center is k * delta
        self.widths.append(width)
        # The value of the weight (h in the article) for the first center is selected randomly
        # The dimension of the weight is equal to the dimension of Y (output)
        weight = np.zeros(output_size)
        for i in range(len(weight)):
            weight [i-1] = np.random.uniform(-1, 1)
        self.weights.append(weight)
        epoch_mse_errors = []  # keeping the amount of error value in every epoch
        numbers_of_centers = []
        GradStep_PerEpoch=[]
        l_NewC = []
        epoch = 1
        while epoch <= num_epochs:
        #for epoch in range(num_epochs):

            epoch_mse_error = 0  # Accumulate the error for each epoch
            input_number = len(input_vectors) # Obtaining the amount of input data
            Delta_Ins_NewC = []
            Ins_NewC = 0
            GradStep_BefIns = 0
            L_GradStep_BefIns = []

            for i in range(input_number):
                # The index of list starts from zero but index 0 is used before "for loop" as the first items and default values
                # Then from the second item in input_vector which has index of 1 will be considered
                input_vector = input_vectors[i]
                target_output = target_outputs[i]
                # Calculating x according to the Gaussian Formula: zj = sigma (cjk - Ik ) ** 2 and xj = exp ( - zj / widthj ** 2)
                # Because x is a list that keeps the distance of current input data from every center in the list of centers.
                # Then, for every center in the list of centers, x should be calculated
                x = np.zeros(len(self.centers))
                for j in range(len(self.centers)):
                    x[j-1] = np.exp((-1 * np.sum((self.centers[j-1] - input_vector) ** 2))/ (self.widths[j-1] ** 2))
                # Calculating output according to the x, weight (h in the formula) and gamma
                # The formula is: y(output) = sigma(hj(weight)*xj) + gamma
                hx = np.zeros(len(target_output))
                for j in range(len(self.centers)):
                    # Because the index of list starts from zero but j starts from 1, the indexes of variables are j-1
                    xj = x[j-1]
                    hj = self.weights[j-1]
                    hj = hj * xj
                    hx += hj
                output = hx + self.gamma
                error = target_output - output # Calculating error for current input
                # Calculating minimum distance of current input from every center in the list of centers
                d1 = np.zeros(len(self.centers))
                # np.linalg.norm calculates the Euclidean norm of the vector
                for j in range(len(self.centers)):
                    d1[j-1] = np.linalg.norm(self.centers[j-1] - input_vector)
                distance = np.min(d1)
                # Evaluating the condition of creating new center, weight and width
                if (np.linalg.norm(error) > epsilon) and (distance > delta):
                   #print ("The condition that insert new center is TRUE and ")
                   Ins_NewC = Ins_NewC + 1
                   Delta_Ins_NewC.append(delta)
                   #print("The Number of Gradient Step Before Insertion is " + str(GradStep_BefIns))
                   L_GradStep_BefIns.append(GradStep_BefIns)
                   # Creating new center
                   center = input_vector
                   self.centers.append(center)
                   # Creating new weight according to creaed center
                   weight = error
                   self.weights.append(weight)
                   # Creating new width according to creaed center
                   width = k * distance
                   self.widths.append(width)
                else:
                   GradStep_BefIns = GradStep_BefIns + 1

                   # if the condition of creating new center was not true, the gradient descent method should be run
                   self.gradient_descent(input_vector, x, error, alpha)

                if delta > delta_min:
                   delta = delta * np.exp(-1 / tau)
                # Converting error vector to a scaler number in oder to giving it to the plot
                mse_error = np.sqrt(np.sum(error ** 2))
                epoch_mse_error += mse_error  # Accumulate the squared error
            epoch_mse_error /= len(input_vectors)  # Calculate the MSE for the epoch
            epoch_mse_errors.append(epoch_mse_error)
            print('The amount of error in ' + f'Epoch {epoch}' + ' is: ' + str(epoch_mse_error))
            l_NewC.append(Ins_NewC)
            print('The Number of New Centers in ' + f'Epoch {epoch}' + ' is: ' + str(Ins_NewC))
            print('The Number of going through Gradient Descent in ' + f'Epoch {epoch}' + ' is: ' + str(GradStep_BefIns))
            GradStep_PerEpoch.append(GradStep_BefIns)
            numbers_of_centers.append(len(self.centers))
            print('The number of centers in '+ f'Epoch {epoch}' + ' is: ' + str(len(self.centers)))
            print('The Centers in '+ f'Epoch {epoch}' + ' are as follows: ')
            print(self.centers)
            centers_matrix = np.array(self.centers)
            # Print Scatter Plot of centers and Input Data
            plt.scatter(input_vectors[:, 2], input_vectors[:, 3], label='Input Data', marker='o', edgecolors='blue', facecolors='none')
            plt.scatter(centers_matrix[:, 2], centers_matrix[:, 3], marker='X', color='red', label='Centers')
            plt.xlabel('Feature 3')
            plt.ylabel('Feature 4')
            plt.title(f'Epoch {epoch}')
            plt.legend()
            plt.show()
            #Print the amount of Delta in Insertion New Centers in this epoch
            #print("The amount of Delta in Insertion New Centers in " + f'Epoch {epoch + 1}' + " is: ")
            #print(Delta_Ins_NewC)
            # Break the loop if the error is less than 0.2
            if epoch_mse_error < error_threshold:
                break

            if epoch < (num_epochs + 1):
                epoch += 1

        #Plot the errors
        # Normalize GradStep_PerEpoch
        # Define the desired range for normalization
        if epoch == 101:
            epoch = epoch - 1
        max_mse_error = max(epoch_mse_errors)
        min_mse_error = min(epoch_mse_errors)
        # Find the min and max values of GradStep_PerEpoch
        min_gradstep_value = min(GradStep_PerEpoch)
        max_gradstep_value = max(GradStep_PerEpoch)
        # Normalize the values to the desired range
        RangeOfGradStep = max_gradstep_value - min_gradstep_value
        rangeOfMse = max_mse_error - min_mse_error
        normalized_gradstep = [min_mse_error + ((value - min_gradstep_value) / (RangeOfGradStep)) * (rangeOfMse) for value in GradStep_PerEpoch]

        plt.plot(range(epoch), epoch_mse_errors, label='MSE Errors')
        plt.plot(range(epoch), normalized_gradstep, label='Normalized GradStep')
        plt.xlabel('Epoch')
        plt.ylabel('value')
        plt.title('MSE and Normalized Numbers of Gradient step in everyepoch vs Epoch')
        plt.legend()
        plt.show()

        plt.plot(range(epoch), numbers_of_centers, label='Number of Centers in every epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Centers')
        plt.title('Number of Centers in every epoch vs Epoch')
        plt.legend()
        plt.show()

        print("The Number of going through Gradient Descent in all epochs are: ")
        print(GradStep_PerEpoch)
        print("The Number of New Centers in all epochs are: ")
        print(l_NewC)
        print("The number of centers are: " + str(len(self.centers)))
        print("The Centers at the end of algorithm are as follows: ")
        print(self.centers)
        print("The amount of errors are as follows: ")
        print(epoch_mse_errors)

    def predict(self, input_vectors):
        output_vector = []
        for i in range(len(input_vectors)):
            input_vector = input_vectors[i]
            # Calculating x according to the Gaussian Formula: zj = sigma (cjk - Ik ) ** 2 and xj = exp ( - zj / widthj ** 2)
            # Because x is a list that keeps the distance of current input data from every center in the list of centers.
            # Then, for every center in the list of centers, x should be calculated
            x = np.zeros(len(self.centers))
            for j in range(len(self.centers)):
                x[j - 1] = np.exp((-1 * np.sum((self.centers[j - 1] - input_vector) ** 2)) / (self.widths[j - 1] ** 2))
            # Calculating output according to the x, weight (h in the formula) and gamma
            # The formula is: y(output) = sigma(hj(weight)*xj) + gamma
            hx = np.zeros(self.output_size)
            for j in range(len(self.centers)):
                xj = x[j - 1]
                hj = self.weights[j - 1]
                hj = hj * xj
                hx += hj
            output = hx + self.gamma
            output_vector.append(output)
        return np.argmax(output_vector, axis=1)

# Usage with Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Split Iris datast to train and test sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# One-hot encode the target outputs:
# OneHotEncoder is used for converting categorical integer features (like class labels) into a one-hot encoded representation.
# For example in Iris dataset we have 3 categorical data then (0, 1, 0) shows category No. 2
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
# Create and train the RAN
input_size = X_train.shape[1]
#input_size which is shape[1] of Iris dataset is the number of columns(feaures) in Iris dataset that is 2
output_size = y_train_encoded.shape[1]
#output_size which is shape[1] of one-hot encoded is the number of columns(class labels) in Iris dataset that is 3
#According to the article delta_max is the largest length of the entire input space of non_zero probability density
delta_max = np.max(np.linalg.norm(X_train, axis=1))
#Create an object of ResourceAllocatingNetwork.
ran = ResourceAllocatingNetwork(input_size, output_size)
#Epsilon is a desired accuracy of output of the Network.
#epsilon = 0.02
epsilon = 0.5
# The distance delta shrinks until it raches delta_min which is a smallest length.
# the network will average over features that are smaller than delta_min.
#delta_min = 0.07
delta_min = 0.9
# k is an overlap factor. As k grows larger, the responses of the units overlap more nd more.
#k = 0.87
k = 0.5
# tau is a decay constant
#tau = 17
tau = 30
# alpha is learning rate
#alpha = 0.02
alpha = float(input("Enter the learning rate (alpha): "))
num_epochs = 100
#error_threshold = 0.2
error_threshold = float(input("Enter the Error Threshold (error_threshold): "))
ran.train(X_train, y_train_encoded, epsilon, delta_max, delta_min, k, tau, alpha, num_epochs, error_threshold)
# Predict Y values of X_test
y_pred = ran.predict(X_test)

# Evaluate accuracy of Ys that were predicted
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")