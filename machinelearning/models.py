from torch import no_grad, stack, norm
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones, randn
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()


        weight_vector = ones(1, dimensions)
        self.w = Parameter(weight_vector)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        score = tensordot(x, self.w)
        return score

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        if score >= 0:
            return 1
        else: return -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            converge = False
            while not converge:
                converge = True
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    prediction = self.get_prediction(x)
                    if prediction != y:
                        self.w += y * x  # 通过更改自身权重进行计算
                        converge = False



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        # begin with only one hidden layer
        # improve by designing two hidden layers
        super().__init__()
        self.learning_rate = 0.005
        self.layer1 = Linear(1, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, 128)
        self.layer4 = Linear(128, 1)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # create Linear
        output1 = self.layer1(x)
        intput2 = relu(output1)
        output2 = self.layer2(intput2)
        input3 = relu(output2)
        output3 = self.layer3(input3)
        input4 = relu(output3)
        output4 = self.layer4(input4)
        return output4
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        prediction = self.forward(x)
        return mse_loss(prediction, y)

 
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        lossRate = 1
        # optimizer -- gradient
        # total loss :0.000182
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        step = 0
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        while step <= 1500:
            step += 1
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.get_loss(batch['x'], batch['label'])
                # 后向传播
                loss.backward()
                optimizer.step()




class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        self.learning_rate = 0.005
        self.layer1 = Linear(input_size, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, 128)
        self.layer4 = Linear(128, output_size)



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # create Linear
        output1 = self.layer1(x)
        intput2 = relu(output1)
        output2 = self.layer2(intput2)
        input3 = relu(output2)
        output3 = self.layer3(input3)
        input4 = relu(output3)
        output4 = self.layer4(input4)
        return output4


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        prediction = self.run(x)
        return cross_entropy(prediction, y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        step = 0
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        #accuracy = dataset.get_validation_accuracy()
        step = 0
        while step < 10:
            step += 1
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.get_loss(batch['x'], batch['label'])
                # 后向传播
                loss.backward()
                optimizer.step()



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        # Initialize your model parameters here
        # RNN only  two layers
        self.layerNum = 128
        self.layer1 = Linear(self.num_chars, self.layerNum)
        self.layer2 = Linear(self.layerNum, self.layerNum)
        self.layer3 = Linear(self.layerNum, len(self.languages))
        self.learning_rate = 0.005

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h = tensor([0.0] * self.layerNum)
        for ch in xs:
            h = relu(self.layer1(ch) + h)
        output = self.layer2(h)
        input = relu(output)
        output1 = self.layer3(input)
        return output1

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return cross_entropy(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        step = 0
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        while step < 10:
            step += 1
            for batch in dataloader:
                optimizer.zero_grad()
                moveBatch = movedim(batch['x'], 0, 1)
                loss = self.get_loss(moveBatch, batch['label'])
                # 后向传播
                loss.backward()
                optimizer.step()


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    # 卷积神经网络
    Output_Tensor = empty(
        (input_tensor_dimensions[0] - weight_dimensions[0] + 1, input_tensor_dimensions[1] - weight_dimensions[1] + 1))
    for i in range(input_tensor_dimensions[0] - weight_dimensions[0] + 1):
        for j in range(input_tensor_dimensions[1] - weight_dimensions[1] + 1):
            Output_Tensor[i, j] = (input[i:i + weight_dimensions[0], j:j + weight_dimensions[1]] * weight).sum()
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        input_size = 26 * 26
        output_size = 10
        self.learning_rate = 0.005
        self.layer1 = Linear(input_size, 256)
        # self.layer2 = Linear(256, 128)
        # self.layer3 = Linear(128, 128)
        self.layer4 = Linear(256, output_size)


    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        print(x.size())
        x = x.reshape(len(x), 28, 28)
        print(x.size())
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        print(x.size())
        x = x.flatten(start_dim=1)
        print(x.size())
        output1 = self.layer1(x)
        input2 = relu(output1)
        # output2 = self.layer2(input2)
        # input3 = relu(output2)
        # output3 = self.layer3(input3)
        # input4 = relu(output3)
        output4 = self.layer4(input2)
        return output4

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        prediction = self.run(x)
        return cross_entropy(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        step = 0
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        #accuracy = dataset.get_validation_accuracy()
        step = 0
        while step < 10:
            step += 1
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.get_loss(batch['x'], batch['label'])
                # 后向传播
                loss.backward()
                optimizer.step()
 