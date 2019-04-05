#!/bin/bash

cp experiments/test_efficiency.py ./
mkdir experiments/logs

python test_efficiency.py 1 &
pid1=$!
python test_efficiency.py 2 &
pid2=$!

python test_efficiency.py 0

kill $pid1 $pid2

rm test_efficiency.py

read -p "Remove all logs? y/[n]" cmd
if [ "$cmd" == "y" ]; then
	rm -r experiments/logs
fi
