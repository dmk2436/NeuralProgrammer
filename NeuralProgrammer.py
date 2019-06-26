import torch
import torch.nn as nn

class NeuralProgrammer():
    
    embedding_size = 10 # word representation dimension
    # change the value according to your custom embedding algorithm applied
    
    columns = 10
    rows = 10
    # change this, too, as your preference

row_select = []
scalar_output = []
lookup_output = []

def __init__(self):
    init(self)
    return

def init(self):
    self.timestep = 0
    self.T = -1
    self.train = True
    
    self.question = Question()
    self.table = Table()
    self.questionRNN = QuestionRNN()
    self.historyRNN = HistoryRNN()
    self.operationSelector = Selector()
    self.dataSelector = Selector()
    self.operation = Operation()
    return

def setTrainMode(boolean):
    self.train = boolean
    return

class Question:
    
    size = -1
    textSize = -1
    words = []
    representation = []
    
    def __init__(self):
        return
    
    def inputSentence(self, sentence):
        if(sentence[len(sentence) - 1] == '.'):
            sentence = sentence[:-1]
        words = sentence.split(' ')
        self.size = len(words)
        texts = []
        numbers = []
        for i in range(self.size):
            if(words[i].isDigit()):
                numbers = [numbers, words[i]]
            else:
                texts = [texts, words[i]]
        self.textSize = len(texts)
        self.words = texts.extend(numbers)
        for i in range(self.size):
            self.representation = [self.representation, represent(self.words[i])]
        return
    
    def represent(word):
        representation = ''
        # add your custom embedding algorithm (e.g. Word2vec, GloVe)
        return representation
    
    def getRepresentation(index):
        return self.representation[index]

class Table:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.table = torch.zeros((columns, rows))
        return
    
    def getTable():
        return self.table

class QuestionRNN(nn.Module):
    def __init__(self, embedding_size):
        super(QuestionRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden = self.initHidden()
        self.i2h = nn.Linear(embedding_size * 2, embedding_size)
        return

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 0)
        hidden = self.i2h(combined)
        self.hidden = nn.Tanh(hidden)
        return
    
    def getHidden(self):
        return self.hidden

    def initHidden(self):
        return torch.zeros(self.embedding_size)

class HistoryRNN(nn.Module):
    def __init__(self, embedding_size):
        super(HistoryRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden = self.initHidden()
        self.i2h = nn.Linear(embedding_size * 3, embedding_size)
        return

    def forward(self, operation_intermediate, operation_probability, column_intermediate, column_probability):
        operation_intermediate = torch.matmul(
            transpose(operation_probability),
            operation_representation)
        column_intermediate = torch.matmul(
            torch.transpose(column_probability),
            column_representation)
        combined = torch.cat((operation_intermediate, column_intermediate), 0)
        hidden = self.i2h(combined)
        self.hidden = nn.Tanh(hidden)
        return
        
    def getHidden(self):
        return self.hidden

    def initHidden(self):
        return torch.zeros(self.embedding_size)

class Selector(nn.Module):
    def __init__(self, embedding_size, num):
        super(Selector, self).__init__()
        
        self.embedding_size = embedding_size
        self.num = num
        self.matrix = self.initMatrix()
        self.intermediate = nn.Linear(embedding_size, num)
        return
    
    def process(self, question, history, representation):
        porbability = torch.Softmax(
            torch.matmul(representation,
                         torch.Tanh(
                         torch.cat((question, hidden), 0))))
        return probability
    
    def initMatrix(self):
        return torch.zeros((self.embedding_size, self.num))

class Operation:
    size = 9
    
    def __init__(self):
        return
    
    def Sum():
        return torch.sum(
            torch.matmul(row_select[timestep - 1], table))
    
    def Count():
        return torch.sum(row_select[timestep - 1])
    
    def Difference():
        return scalar_output[timestep - 3] - scalar_output[timestep - 1]
    
    def Greater():
        return
    
    def Lesser():
        return
    
    def And():
        _and = torch.min(
            torch.cat(row_select[timestep - 1], row_select[timestep - 2]),
            0)
        return _and
    
    def Or():
        _or = torch.max(
            torch.cat(row_select[timestep - 1], row_select[timestep - 2]),
            0)
        return _or
    
    def Assign():
        _row_select = row_select[timestep - 1].clone()
        _row_select = torch.reshape([_row_select.shape, 1])
        assign = torch.Tensor()
        for i in range(table.columns)
            assign = torch.cat(assign, _row_select)
        return assign
    
    def Reset():
        return torch.zeros(table.rows)

def getOutput():
    scalar_answer = torch.matmul(alpha_operation,
                                 torch.add(operation.Count(), operation.Difference(),
                                           torch.matmul(alpha_column, operation.Sum())))
    lookup_answer = torch.matmul(alpha_operation,
                                 torch.mul(operation.Assign(), alpha_column))
    return scalar_answer, lookup_answer

def getRowSelect():
    row_select = torch.matmul(alpha_operation,
                              torch.add(operation.Add(), operation.Or(), operation.Reset(),
                                        torch.matmul(alpha_column,
                                                    torch.add(
                                                        torch.add(operation.Greater(), operation.Lesser())[K, C],
                                                        getTextMatch()[0, K]))))
    return row_select

def train():
    return

def test():
    return
