#!/bin/bash

mv -v ../*.png $1
mv -v ../*.h5 $1
mv -v ../*.pkl $1
cp -v ../train_transformer_unsupervised.py $1
cp -v ../train_transformer.py $1
cp -v ../train_transformer_random_init.py $1
