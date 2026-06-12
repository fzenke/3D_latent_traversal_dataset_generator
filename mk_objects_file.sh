#!/bin/bash


N=all # Number of objects per synset use "all" for all

ARGS=""
for SYNSETID in 02691156 02876657 02954340 03001627 03261776 03467517 03636649 03642806 03790512 03797390 03928116 04401088; do
ARGS="$ARGS --synset $SYNSETID $N"
done

echo $ARGS

python sample_objects.py --models-path /tachyon/groups/gzenke/datasets/ShapeNetCoreV2 --output my_objects.npy $ARGS
