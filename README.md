brae
====

Bilingual contrained phrase embedding for MT

Monoligual phrase embedding works with backpropagation

$ time ./phraseEmbedding.py
Reading vectors in binary format
Read 555587 entries from the binary file
Embedding shape = (50,)
Cost (0) = 1144.22855732
=================================
Iteration : 1
Cost (1) = 571.042721099
=================================
Iteration : 2
Cost (2) = 380.347088307
=================================
Iteration : 3
Cost (3) = 285.130000339
=================================
Iteration : 4
Cost (4) = 228.040957773
=================================
Iteration : 5
Cost (5) = 190.001100333
=================================
Iteration : 6
Cost (6) = 162.849781705
=================================
Iteration : 7
Cost (7) = 142.478591007
=================================
Iteration : 8
Cost (8) = 126.64131166
=================================
Iteration : 9
Cost (9) = 113.968241981
=================================

This experiment had 
\alpha (Learning rate) = 0.01
\lambda (Regularization parameter) = 0.1
10 iterations of batch backpropagation were performed

TODO:
1. Print regularized cost
2. There seem to be some -1s in the input vectors, where are these coming from ? 
3. Create a class for phraseEmbedding
4. Do this for two languages
5. Run this for multiple settings of n, \lambda and \alpha
6. Implement fine-tuning (BRAE) mechanism
