"""

A wrapper for reading and interpreting vectors pre-trained using word2vec
Derived from Google's word2vec python wrapper
https://pypi.python.org/pypi/word2vec

"""

import numpy as np

class WordVectors:
  # The vocabulary 
  vocab = None
  # Trained word embeddings as fixed dimensional vectors
  vectors = None
  l2Norm = None

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
      self.from_binary(fname)
      self.l2norm = np.vstack(self.unitvec(vec) for vec in self.vectors)
      print "Read " + str(len(self.vocab)) + " entries from the binary file"
      print "Embedding shape = " + str(np.shape(self.vectors[0]))
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
      vocab = []

      vectors = np.empty((vocab_size, vector_size), dtype=np.float)
      binary_len = np.dtype(np.float32).itemsize * vector_size
      for line_number in xrange(vocab_size):
        # mixed text and binary: read text first, then binary
        word = ''
        while True:
          ch = fin.read(1)
          if ch == ' ':
            break
          word += ch
        vocab.append(word)

        vector = np.fromstring(fin.read(binary_len), np.float32)
        vectors[line_number] = vector
        fin.read(1)  # newline

    self.vocab = np.array(vocab)
    self.vectors = vectors

  def ix(self, word):
    """
    Returns the index on self.vocab and self.l2norm for `word`
    """
    temp = np.where(self.vocab == word.encode('UTF-8'))[0]
    if temp.size == 0:
      return -1
      #raise KeyError('Word not in vocabulary')
    else:
      return temp[0]


  def __getitem__(self, word):
    return self.get_vector(word)


  def get_vector(self, word):
    """
    Returns the (l2norm) vector for `word` in the vocabulary
    """
    idx = self.ix(word)
    if idx == -1:
      # Key not found
      return None
    else:
      return self.l2norm[idx]

  def unitvec(self, vec):
    return (1.0 / np.linalg.norm(vec, ord=2)) * vec
