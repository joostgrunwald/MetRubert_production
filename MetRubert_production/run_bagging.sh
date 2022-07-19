#!/bin/bash

for n in {2..17}
do
	for i in $(seq 0 $n)
	do
        	echo "Running bagging for index $i and num bagging $n"
        	nice -19 python3 main_dutch.py --bagging_index $i --num_bagging $n
	done
done
