#!/usr/bin/env bash

gigawordSubset=afp
annotatedGW=/export/a04/gkumar/code/custom/brae/data/LDC2012T21

if [ ! -e $annotatedGW ]; then
  echo "Annotated Gigaword not found at $annotatedGW"
  exit 1
fi

tools=$annotatedGW/tools/agiga_1.0

if [ ! -e $tools ]; then
  echo "GW tools not found at $tools"
  exit 1;
fi

outDir=en_gigaword_$gigawordSubset
mkdir -p $outDir

# Now find the files that need to be processed
for file in `find $annotatedGW/data -iname "${gigawordSubset}_*.xml.gz"`; do
  fileName=`basename $file`
  java -cp $tools/build/agiga-1.0.jar:$tools/lib/* \
    edu.jhu.agiga.AgigaPrinter words \
    $file > $outDir/${fileName%%.*}.en.tok
done


# java -cp build/agiga-1.0.jar:lib/* edu.jhu.agiga.AgigaPrinter words /export/corpora/LDC/LDC2012T21/data/xml/afp_eng_199405.xml.gz


