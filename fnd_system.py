import pandas as pd
import numpy as np
import torch
import nltk
import re
import string
import torch.nn as nn
import torch.nn.functional as F
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import joblib
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


nltk.download('stopwords')
nltk.download('wordnet')


""" Model to Training Data

  Forward Function
Take length of Embedding and Dim of it and Create Embedding Layer by torch Framework
Create Neural Network Layer take 903 input and return 256 as output layer
then bass output to non-activation function layer then add dropout to output
bass output to Linear take 256 and return number of class 1 and then add sigmoid to output

  Conv Function
Take inputs ( self, inputs layer[batch size of training inputs*Featuers] (64,1440) )

conv inputs layer from [64,1440] to [64,903]

first 300s number repersent the vector sentence(Statement) of Embedding

301 add Similarity of Statement Featuer (first [410] numbers of orignal input) and
  Subject Featuer (second [30] numbers of orignal input)

from 302 to 602 add number repersent the vector sentence(Subject) of Embedding

602 add Similarity of Subject Featuer ([30] numbers of orignal input) and
  Justification Featuer (second [1000] numbers of orignal input)

from 602 to 902 add number repersent the vector sentence(Justifaction) of Embedding

903 add Similarity of Justification Featuer ([100] numbers of orignal input) and
  Statement Featuer (second [410] numbers of orignal input)

and return [64,903]

Then Pass output of Conv to Forward


"""

class NN(nn.Module):


  def __init__(self, vocab_size,batch_size, embedding_dim,lens,word_embedding,hidden_dim):

    super(NN, self).__init__()

    self.device='cuda' if torch.cuda.is_available() else 'cpu'
    self.embedding_dim=embedding_dim
    self.hidden_dim=hidden_dim
    self.batch_size=batch_size
    self.lens=lens
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.embedding.weight.data.copy_(word_embedding)

    self.lstm = nn.LSTM(3*self.embedding_dim+3, self.hidden_dim, 2, dropout=0.7, batch_first=True)
    self.fc = nn.Linear(self.hidden_dim,1)
    self.dropout=nn.Dropout(0.7)
    self.sig = nn.Sigmoid()



  def forward(self, x,hidden):

    batch = x.size(0)

    x = self.conv(x)


    lstm_out, hidden = self.lstm(x, hidden)
    lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)
    out = self.dropout(lstm_out)

    out=self.sig(self.fc(out))
    sig_out = out.view(batch, -1)
    sig_out = sig_out[:, -1]
    return sig_out, hidden


  def conv(self,x):

    bacth=len(x)
    ret=torch.zeros((bacth,3*self.embedding_dim+3)).to(self.device)
    st=torch.zeros(self.embedding_dim).to(self.device)
    su=torch.zeros(self.embedding_dim).to(self.device)
    ju=torch.zeros(self.embedding_dim).to(self.device)



    for i in range(bacth):

      st=self.embedding(x[i][0:self.lens[0]]).sum(dim=0)/(x[i][0:self.lens[0]]!=0).sum()
      su=self.embedding(x[i][self.lens[0]:self.lens[1]]).sum(dim=0)/(x[i][self.lens[0]:self.lens[1]]!=0).sum()
      ju=self.embedding(x[i][self.lens[1]:self.lens[2]]).sum(dim=0)/(x[i][self.lens[1]:self.lens[2]]!=0).sum()

      ret[i][:self.embedding_dim]=st
      ret[i][self.embedding_dim]=F.cosine_similarity(st,su,dim=0)

      ret[i][self.embedding_dim+1:2*self.embedding_dim+1]=su
      ret[i][self.embedding_dim*2+1]=F.cosine_similarity(su,ju,dim=0)

      ret[i][2*self.embedding_dim+2:3*self.embedding_dim+2]=ju

      ret[i][self.embedding_dim*3+2]=F.cosine_similarity(st,ju,dim=0)
    return ret.view(1,bacth,self.embedding_dim*3+3)


  def init_hidden(self):
    weight = next(self.parameters()).data
    hidden = (weight.new(2, 1, self.hidden_dim).zero_().to(self.device),
                  weight.new(2, 1, self.hidden_dim).zero_().to(self.device))
    return hidden

class preprocess():
  def __init__(self):

    super(preprocess, self).__init__()
    self.device='cuda' if torch.cuda.is_available() else 'cpu'
    self.all_statements,self.all_subject,self.all_justifications,self.tags_counts=self.load_data()

    self.word_to_index, self.index_to_word, self.word_to_vec_map =self.read_glove_vecs('dataset/glove.6B.300d.txt')
    self.word_embedding=self.pretrained_embedding_layer(self.word_to_vec_map,self.word_to_index)
    self.model=NN(len(self.word_to_index)+1,32,300,[410,440,1440],torch.from_numpy(self.word_embedding),256)
    self.model.load_state_dict(torch.load('dataset/model.pt',map_location=torch.device(self.device)))
    self.model.to(self.device)

    self.mlb = MultiLabelBinarizer(classes=sorted(self.tags_counts.keys()))

    self.tfidf_vectorizer_model = pickle.load(open('vectorize.sav', 'rb'))

    self.loaded_model = pickle.load(open('model.sav', 'rb'))



  """
     Clean Text

  """

  def load_data(self):
    cols=['index','ID','label','statement','subject','speaker','speaker_job','state','party','barely_true',
        'false','half_true','mostly_true','pants_on_fire','context','justification']

    dftrain=pd.read_csv("dataset/train.tsv",sep="\t",header=None,names=cols)
    dfval=pd.read_csv("dataset/val.tsv",sep="\t",header=None,names=cols)
    dftest=pd.read_csv("dataset/test.tsv",sep="\t",header=None,names=cols)

    dftrain=dftrain.loc[:,['statement','justification','subject']]
    dfval=dfval.loc[:,['statement','justification','subject']]
    dftest=dftest.loc[:,['statement','justification','subject']]

    dftrain=dftrain.append(dfval)

    all_subject = [subs.split(',') for subs in dftrain.dropna(axis=0)['subject'].values]

    dftrain=dftrain.append(dftest)

    dftrain=dftrain.dropna(axis=0)

    all_statements=dftrain['statement'].values
    all_statements=[self.clean(statement,True).split() for statement in all_statements]



    all_justifications=dftrain['justification'].values
    all_justifications=[self.clean(justification,True).split() for justification in all_justifications]

    tags_counts={}

    for tag in all_subject:
        for indtag in tag:
            if indtag not in tags_counts:
                tags_counts[indtag]=1
            else:
                tags_counts[indtag]=tags_counts[indtag]+1

    return all_statements,all_subject,all_justifications,tags_counts


  def clean(self,text,is_quality):
    text=text.lower()
    stp=set(stopwords.words("english"))
    placesp = re.compile('[/(){}\[\]\|@,;]')
    removech= re.compile('[^0-9a-z #+_]')
    st=WordNetLemmatizer()
    text=re.sub(placesp,' ',text)
    text=re.sub(removech,' ',text)

    if is_quality == True:
      return text.translate(str.maketrans("", "", string.punctuation))

    text=text.split()
    text=[w for w in text if not w in stp]
    text=[st.lemmatize(w) for w in text]
    text=" ".join(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
    """
  Transfer sentence to indeces word in Embedding
  take text and word to index dictionary
  return list of indeces word in Embedding

  """
  def transfer_sent(self,text,word_to_index):
    text=text.split(' ')
    ret=[]
    for w in text:
      if w in word_to_index and w !="":
        ret.append(word_to_index[w])
    return ret


  def padding_test(self,text,ln):
    if len(text)<ln:
      text+=[0]*(ln-len(text))
    return text

  """
  Word Embeddings of words take dictionary of word to embedding and word to index
  and return Embeddings Matrix [index,Embedding]

  """

  def pretrained_embedding_layer(self,word_to_vec_map, word_to_index):
      vocab_len = len(word_to_index) + 1
      emb_matrix = np.zeros((vocab_len,300))
      for word, index in word_to_index.items():
          emb_matrix[index, :] = word_to_vec_map[word]
      return emb_matrix

  """
    Read Glove File take url of file return the two dictionaries ( word to index and word to vector in embedding )
    and one list of index to word
    (glove file url) --> words_to_index, index_to_words, word_to_vec_map

  """
  def read_glove_vecs(self,glove_file):
        with open(glove_file, 'r',encoding='UTF-8') as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            i = 1
            words_to_index = {}
            index_to_words = {}
            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1
        return words_to_index, index_to_words, word_to_vec_map


  def quality(self,statement,justification):

    smoother = SmoothingFunction()
    score_statement = sentence_bleu(self.all_statements, statement, smoothing_function=smoother.method2)
    score_justification = sentence_bleu(self.all_justifications, justification,smoothing_function=smoother.method2)
    return (score_statement+score_justification)/2


  def reality(self,statement,subject,justification):

    quality_new = self.quality(self.clean(statement,True),self.clean(justification,True))

    statement=self.padding_test(self.transfer_sent(self.clean(statement,False),self.word_to_index),410)
    subject=self.padding_test(self.transfer_sent(self.clean(subject,False),self.word_to_index),30)
    justification=self.padding_test(self.transfer_sent(self.clean(justification,False),self.word_to_index),1000)

    once=torch.tensor([statement+subject+justification]).to(self.device)
    self.model.eval()
    h=self.model.init_hidden()
    pred,h=self.model(once,h)
    return pred.item()*100,quality_new*100

  def text_prepare(self,text):

      text =text.lower()
      text = re.sub(re.compile('[/(){}\[\]\|@,;]'),' ',text)
      text = re.sub(re.compile('[^0-9a-z #+_]'),'',text)
      text = ' '.join(w for w in text.split() if w not in set(stopwords.words('english')))
      return text

  def detect_subject(self,text):

    text=self.text_prepare(text)
    text = self.tfidf_vectorizer_model.transform([text])
    ytrain = self.mlb.fit_transform(self.all_subject)

    ypred = self.loaded_model.predict(text)
    ypred_inversed = self.mlb.inverse_transform(ypred)

    return ypred_inversed[0]


model = preprocess()

def get_subject(statement):
    return model.detect_subject(statement)

# in_all = model.reality(statement,subject,justification)
# print("The credibility of new {:.2f}% and the quality {:.2f}%".format(in_all[0],in_all[1]))

def detect(statement, subject, justification):
    in_all = model.reality(statement,subject,justification)
    print("The credibility of new {:.2f}% and the quality {:.2f}%".format(in_all[0],in_all[1]))
    return in_all
