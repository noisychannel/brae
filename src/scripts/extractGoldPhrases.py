#!/usr/bin/env python

'''
Extracts gold standard phrases from a Moses output file from decoding
which contains phrase alignment information. 
Ideally, this output should have been created via forced (constrained)
decoding. 
'''

import codecs
import sys
import re

srcFile = codecs.open(sys.argv[1], encoding="utf8")
tgtFile = codecs.open(sys.argv[2], encoding="utf8")
ppFile = codecs.open(sys.argv[3], "w+", encoding="utf8")

phrasePairs = set()

srcSentences = []

# First read the source file and store the tokenized sentences
for line in srcFile:
  line = line.strip()
  srcSentences.append(line.split())

sentIndex = 0

# Now read the target file and extract alignment information 
for line in tgtFile:
  line = line.strip()
  if line != "":
    lineComp = line.split()
    currentSrc = srcSentences[sentIndex]
    currentTgtPhrase = []
    for word in lineComp:
      alignmentMatch = re.search("\|(\d)-(\d)\|", word)
      if alignmentMatch:
        # This token contains phrase alignment information
        phraseStart = int(alignmentMatch.group(1))
        phraseEnd = int(alignmentMatch.group(2))
        # Extract this phrase from the source
        srcPhrase = " ".join(currentSrc[phraseStart:phraseEnd+1])
        tgtPhrase = " ".join(currentTgtPhrase)
        # Adding to a set ensures we have no duplicates
        phrasePairs.add((srcPhrase, tgtPhrase))
        currentTgtPhrase = []
      else:
        # Regular token
        currentTgtPhrase.append(word)
  sentIndex = sentIndex + 1

# Write out all the phrase pairs that were observed
for phrasePair in phrasePairs:
  ppFile.write(phrasePair[0] + "\t" + phrasePair[1] + "\n")

ppFile.close()
tgtFile.close()
srcFile.close()
