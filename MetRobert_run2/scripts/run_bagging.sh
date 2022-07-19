#!/bin/bash

INDEXES=$(seq 0 9)
for i in $INDEXES
do
        echo "Running bagging for index $i"
        nice -19 python3 main_dutch.py --bagging_index $i
done
