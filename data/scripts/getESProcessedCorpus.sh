#!/usr/bin/env bash

gigawordSubset=afp
annotatedGW=/export/a04/gkumar/corpora/gigaword-spanish/afp
JOSHUA=/export/a04/gkumar/code/joshua

if [ ! -e $annotatedGW ]; then
  echo "Spanish Gigaword not found at $annotatedGW"
  exit 1
fi

outDir=es_gigaword_$gigawordSubset
mkdir -p $outDir

# Now find the files that need to be processed
for file in `find $annotatedGW -iname "${gigawordSubset}_*.words"`; do
  $JOSHUA/scripts/training/penn-treebank-tokenizer.perl -l es < $file >> $outDir/all_$gigawordSubset.tok.es
done


# java -cp build/agiga-1.0.jar:lib/* edu.jhu.agiga.AgigaPrinter words /export/corpora/LDC/LDC2012T21/data/xml/afp_eng_199405.xml.gz


