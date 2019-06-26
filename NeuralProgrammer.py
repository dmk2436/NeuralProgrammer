import torch
import torch.nn as nn
import torch.optim as optim

embedding_size = 10 # word representation dimension
# change the value according to your custom embedding algorithm applied

columns = 10
rows = 10

_lambda = 0.1
lr = 1e-1
n_epochs = 1000
# change these, too, as your preference

def represent(self, word):
    return 0.314
# add your embedding algorithm (e.g. Word2vec, GloVe)

row_select = torch.zeros(table.rows)
scalar_output = 0.0
lookup_output = torch.zeros((table.rows, table.columns))

question_hiddens = torch.Tensor()

operation_probability = torch.zeros(operation.size)
column_probability = torch.zeros(table.columns)

pivot_g = 0.0
pivot_l = 0.0

timestep = 1
T = -1
train = False

question = Question()
table = Table(columns, rows)
questionRNN = QuestionRNN(embedding_size)
historyRNN = HistoryRNN(embedding_size)
operation = Operation()
operationSelector = Selector(embedding_size, operation.size)
dataSelector = Selector(embedding_size, table.columns)

operation_representation = torch.zeros(operation.size, embedding_size)
column_representation = torch.zeros(table.columns, embedding_size)

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
            if(words[i].isdigit()):
                numbers = [numbers, words[i]]
            else:
                texts = [texts, words[i]]
        self.textSize = len(texts)
        self.words = texts.extend(numbers)
        for i in range(self.size):
            self.representation = [self.representation, represent(self.words[i])]
        return
    
    def getRepresentation(self):
        return self.representation

class Table:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.table = torch.zeros((columns, rows))
        return
    
    def setTable(self, _table):
        self.table = torch.from_numpy(_table)
        return
    
    def getTable(self):
        return self.table

class QuestionRNN(nn.Module):
    def __init__(self, embedding_size):
        super(QuestionRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden = self.initHidden()
        self.i2h = nn.Linear(embedding_size * 2, embedding_size)
        return

    def forward(self, _input):
        combined = torch.cat((_input, self.hidden), 0)
        hidden = self.i2h(combined)
        self.hidden = nn.Tanh(hidden)
        question_hiddens = tf.concat(question_hiddens, self.hidden, 1)
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

    def forward(self, operation_probability, column_probability):
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
        self.intermediate = nn.Linear(embedding_size, num)
        return
    
    def process(self, question, representation):
        porbability = torch.Softmax(
            torch.matmul(representation,
                         torch.Tanh(
                         torch.cat((question, hidden), 0))))
        return probability

class Operation:
    size = 9
    
    def __init__(self):
        return
    
    def Sum(self):
        return torch.sum(
            torch.matmul(row_select[timestep - 1], table))
    
    def Count(self):
        return torch.sum(row_select[timestep - 1])
    
    def Difference(self):
        return scalar_output[timestep - 3] - scalar_output[timestep - 1]
    
    def Greater(self):
        return
    
    def Lesser(self):
        return
    
    def And(self):
        _and = torch.min(
            torch.cat(row_select[timestep - 1], row_select[timestep - 2]),
            0)
        return _and
    
    def Or(self):
        _or = torch.max(
            torch.cat(row_select[timestep - 1], row_select[timestep - 2]),
            0)
        return _or
    
    def Assign(self):
        _row_select = row_select[timestep - 1].clone()
        _row_select = torch.reshape([_row_select.shape, 1])
        assign = torch.Tensor()
        for i in range(table.columns):
            assign = torch.cat(assign, _row_select)
        return assign
    
    def Reset(self):
        return torch.zeros(table.rows)

def getOutput():
    scalar_answer = torch.matmul(alpha_operation,
                                 torch.add(operation.Count(), operation.Difference(),
                                           torch.matmul(alpha_column, operation.Sum())))
    lookup_answer = torch.matmul(alpha_operation,
                                 torch.mul(operation.Assign(), alpha_column))
    return scalar_answer, lookup_answer

def getRowSelect():
    
    C = table.columns
    row_select = torch.matmul(alpha_operation,
                              torch.add(operation.Add(), operation.Or(), operation.Reset(),
                                        torch.matmul(alpha_column,
                                                    torch.add(
                                                        torch.add(operation.Greater(), operation.Lesser())[K, C],
                                                        getTextMatch()))))
    return row_select

def getTextMatch():
    K = question.size - question.textSize
    A = table.table[:, :K, :]
    B = torch.sigmoid(
        torch.matmul(
            A,
            torch.transpose(questionRNN.getHidden()[0:K])))
    D = torch.sum(
    torch.mul(B, A,), 0)
    G = torch.mul(questionRNN.getHidden[0:K](), question.getRepresentation()[0:K, :])
    text_match = torch.sigmoid(
        torch.sum(A, G.reshape(1, G.shape[0], G.shape[1]), 2))
    return text_match

def inference(data):
    return _inference(data, False)

def _inference(data, _train):
    prev_train = train
    train = _train
    table.setTable(data.table)
    question.inputSentence(data.question)
    
    scalar_answer = 0
    lookup_answer = torch.zeros(table.rows, table.columns)
    row_select = torch.ones(table.rows)
    historyRNN.initHidden()
    
    for i in range(question.size):
        questionRNN.forward(question.getRepresentation(i))
    
    q = questionRNN.getHidden()
    
    beta_g = torch.softmax(question_hiddens[:, questionRNN.textSize:questionRnn.size] * operation_representation[3])
    pivot_g = q * torch.sum(torch.mul(beta_g, question.representation))
    beta_l = torch.softmax(question_hiddens[:, questionRNN.textSize:questionRnn.size] * operation_representation[3])
    pivot_l = q * torch.sum(torch.mul(beta_g, question.representation))
    
    for i in range(1, T):
        timestep = i
        operation_probability = operationSelector.process(q, operation_representation)
        column_probability = dataSector.process(q, column_representation)
        
        if(timestep > 2):
            _row_select = getRowSelect()
            _scalar_output, _lookup_output = getOutput()
        else:
            _row_select = torch.zeros(table.rows)
            _scalar_output = 0.0
            _lookup_output = torch.zeros((table.rows, table.columns))
        
        row_select = [row_select, _row_select]
        scalar_output = [scalar_output, _scalar_output]
        lookup_output = [lookup_output, _lookup_output]
        historyRNN.forward(operation_probability, column_probability)
    
    if(T > 0):
        scalar_difference = abs(scalar_output[T] - scalar_output[T - 1])
        lookup_difference = abs(torch.sum(lookup_output[T] - lookup_output[T - 1]))
        if(scalar_difference >= lookup_difference):
            output = scalar_output
        else:
            output = lookup_output
    else:
        output = 0.0
    train = past_train
    return output

def train(dataset):
    loss_record = []
    accuracy_record = []
    optimizer = optim.Adam([questionRNN.parameters(), historyRNN.parameters()], lr=lr)
    for epoch in range(n_epochs):
        loss = 0.0
        for i in range(len(dataset)):
            _inference(dataset[i], True)
            if(dataset[i].n == True):
                a = torch.abs(scalar_answer[T] - dataset[i].y)
                if(a <= delta):
                    loss_scalar = a * a / 2
                else:
                    loss_scalar = delta * a - a * a / 2
                loss += loss_scalar
            else:
                loss_lookup = torch.sum(
                    torch.add(
                        torch.mul(dataset[i].y, torch.log(lookup_answer[T])),
                        torch.mul(1 - dataset[i].y, torch.log(1-lookup_answer[T])))) * (-1/(table.columns * table.rows))
                loss += loss_lookup * _lambda
        loss.backward()
        optimizer.step()
        accuacy = test(dataset)
        loss_record = [loss_record, loss]
        accuracy_record = [accuracy_record, accuracy]
    parameters = [questionRNN.parameters(), historyRNN.parameters()]
    final_loss = loss_record[n_epochs - 1]
    final_accuracy = accuracy_record[n_epochs - 1]
    return [loss_record, accuracy_record], [final_loss, final_accuracy], parameters

def test(dataset):
    correct = 0
    for i in range(len(dataset)):
        output = inference(dataset[i])
        if(dataset[i].n == True):
            if(output == dataset[i].y):
                correct = correct + 1
        else:
            if(torch.mean(torch.eq(ouptut, torch.from_numpy(y))) > 0.9):
                correct = correct + 1
    accuracy = correct / len(dataset) * 100
    return accuracy
