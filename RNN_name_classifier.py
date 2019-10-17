from __future__ import unicode_literals, print_function, division
from io import open
import glob

# glob 모듈을 통해 들어온 경로에 있는 txt 파일을 반환해준다.
def findFiles(path):
    return glob.glob(path)

#print(findFiles('data/names/*.txt'))

import unicodedata
import string

# 파이썬을 파이썬 답게.. 대소문자 알파벳을 출력하는 string 모듈. 유니코드 -> 아스키로 변경하기위해.
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 유니 코드 문자열을 일반 ASCII로 변환하는 함수.
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# 위 예시
#print(unicodeToAscii('Ślusàrski'))

# 언어별 이름 목록인 category_lines 사전 만들기위한 딕셔너리 and list.
category_lines = {}
all_categories = []

# 파일을 읽고 라인으로 분리한다. txt 파일을 읽고 read를 통해 전체를 string 으로 받음.
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split()
    return [unicodeToAscii(line) for line in lines]

# 파일을 읽어 category_lines 딕셔너리에 키 값으로 카테고리를, value 값으로 내용을 리스트로 넣어준다.
for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

import torch

# all_letters 로 문자의 주소 (index 위치) 찾기, 예시 "a" = 0, "b" = 1
def letterToIndex(letter):
    return all_letters.find(letter)

# 검증을 위해서 한 문자를 <1 * n_letters> Tensor로 변경하기
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 한 줄(이름)을 <line_length * 1 * n_letters>,
# 또는 문자 벡터의 어레이로 변경하기
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# n_letters = 57 이다.
#print(letterToTensor('J'))
#print(lineToTensor('Jones').size())

# 네트워크 생성
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        #self.h2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        #hidden2 = self.h2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
# n_letters = 57, n_hidden = 128, n_categories = 18
rnn = RNN(n_letters, n_hidden, n_categories)

# input = Variable(lineToTensor('Albert')) # 6, 1, 57
# hidden = Variable(torch.zeros(1, n_hidden))

# output, next_hidden = rnn(input[0], hidden)
# print(output) -> A의 원핫벡터 넣었을 때 어떤 카테고리가 맞을지 output을 확률로 출력.

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i # 카테고리 이름 , 카테고리 인덱스

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    # 랜덤 카테고리 선택
    category = randomChoice(all_categories)
    # 카테고리안 이름도 랜덤으로 선택
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print(category_tensor)
    print('category =', category, '/ line =', line)
print("-------------------------------------")

import torch.optim as optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr = 0.005)
#learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    #rnn.zero_grad()

    # 이름 길이만큼
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    #print('\n\n',output,'\n')
    optimizer.zero_grad()
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    # learning rate를 곱한 파라미터의 경사도를 파라미터 값에 더한다.
    # for p in rnn.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    return output, loss.data

import time
import math

n_iters = 500000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output) # 카테고리 이름, 카테고리 인덱스
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' %
              (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

n_correct = 0
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    if category == guess:
        n_correct += 1
    if i % 1000 == 0:
        print("evaluating = %d%%" % (i / n_confusion * 100) )

print("accuracy =", n_correct/n_confusion)


