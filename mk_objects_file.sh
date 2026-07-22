#!/bin/bash

# This script samples objects from the 12 synset categories we use for TLT12 and stores them to objects.npy

N=all # Number of objects per synset use "all" for all

ARGS=""
for SYNSETID in 02691156 02958343 02954340 03001627 03261776 03467517 04530566 03642806 03790512 03797390 03928116 04401088; do
ARGS="$ARGS --synset $SYNSETID $N"
done

echo $ARGS

python sample_objects.py --models-path /tachyon/groups/gzenke/datasets/ShapeNetCoreV2 --output objects.npy $ARGS


# some synset IDs
# 02691156 airplane
# 02834778 bicycle
# 02876657 bottle
# 02958343 car 
# 02954340 cap
# 03001627 chair
# 03261776 earphone
# 03467517 guitar
# 03636649 lamp
# 03642806 laptop
# 03797390 mug
# 03928116 piano
# 04401088 telephone
# 04530566 watercraft
