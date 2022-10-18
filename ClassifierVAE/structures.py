from typing import NamedTuple, Any

'''
Set of helper structures to reduce bloat and increase readability across the codebase
'''

class Model_Config(NamedTuple):
    num_heads : Any # How many decoder-classifer pairs
    encoder : Any # Encoder function
    decoder : Any # Decoder function
    head : Any # Classifier function
    input : Any # Task specific input spec
    n_class : Any # Number of Classes
    hard : Any # argmax (T) or softmax (F)

class Encoder_Config(NamedTuple):
    n_class : Any 
    n_dist : Any # Number of categorical distributions
    stack : Any # Dense sizes for encoder
    dense_activation : Any # Activation function
    tau : Any # Temperature tf variable

class Decoder_Config(NamedTuple):
    n_class : Any 
    n_dist : Any 
    stack : Any # Dense sizes for decoder
    dense_activation : Any
    tau : Any 

class Head_Config(NamedTuple):
    n_class : Any
    intermediate : Any # Task-specific layers
    stack : Any # Dense sizes for classifier
    dense_activation : Any

class Wrapper_Config(NamedTuple):
    model : Any 
    loss : Any 
    optim : Any 
    epochs : Any 
    temp : Any 
    acc_metric : Any 

class Encoder_Output(NamedTuple):
    logits_y : Any
    p_y : Any # Fixed Prior

class Decoder_Output(NamedTuple):
    recons : Any # Reconstruced x
    gen_y : Any # Generated Logits
    p_x : Any # Distribution over x
    q_y : Any # y Prior

class Model_Output(NamedTuple):
    y_pred : Any # Classifer Output
    p_x : Any # Reconstructed Distribition
    p_y : Any # Fixed Prior
    q_y : Any # Gumbel Prior
    gen_y : Any # Encoder Output

class Test_Results(NamedTuple):
    acc: Any
    f1: Any 
    rec: Any 
    prec: Any

class Dataset(NamedTuple):
    name : Any
    x_train : Any
    x_test : Any 
    y_train : Any 
    y_test : Any