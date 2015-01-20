import numpy as np

class WordVectors:
  # The vocabulary 
  vocab = None
  # Trained word embeddings as fixed dimensional vectors
  vectors = None

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

    Returns
    -------
    WordVectors class
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
        print word
        vocab.append(word)

        vector = np.fromstring(fin.read(binary_len), np.float32)
        vectors[line_number] = vector
        fin.read(1)  # newline

    self.vocab = np.array(vocab)
    self.vectors = vectors
