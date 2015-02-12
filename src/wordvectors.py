"""

A wrapper for reading and interpreting vectors pre-trained using word2vec
Derived from Google's word2vec python wrapper
https://pypi.python.org/pypi/word2vec

"""

import numpy as np
import sys

class WordVectors:
  # Trained word embeddings as fixed dimensional vectors
  vectorHash = None

  def __init__(self, fname, binary=True):
    """
    Constructor for this class
    Reads vectors created by word2vec

    Parameters
    ----------
    fname : path to file

    """
    if binary:
      print "Reading vectors in binary format"
      self.vectorHash = {}
      self.reverseVectorHash = {}
      self.from_binary(fname)
      print "Read " + str(len(self.vectorHash)) + " entries from the binary file"
      print "Embedding shape = " + str(np.shape(self.vectorHash[self.vectorHash.keys()[0]]))
    else:
      raise Exception("Unsupported mode for reading vectors")


  def from_binary(self, fname):
    """
    Create a WordVectors class based on a word2vec binary file

    Parameters
    ----------
    fname : path to file
    save_memory : boolean

    """
    with open(fname) as fin:
      header = fin.readline()
      vocab_size, vector_size = map(int, header.split())

      binary_len = np.dtype(np.float32).itemsize * vector_size
      for line_number in xrange(vocab_size):
        # mixed text and binary: read text first, then binary
        word = ''
        while True:
          ch = fin.read(1)
          if ch == ' ':
            break
          word += ch

        vector = np.fromstring(fin.read(binary_len), np.float32)
        self.vectorHash[word.decode('utf8')] = vector
        fin.read(1)  # newline


  def __getitem__(self, word):
    return self.get_vector(word)


  def get_vector(self, word):
    if word in self.vectorHash:
      return self.vectorHash[word]
    else:
      return None
