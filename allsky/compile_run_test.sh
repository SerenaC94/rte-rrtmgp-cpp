#!/bin/bash
#argument 1: 0: use cuda, 1: use serial cpu, 2: use parallel cpu

#compile
cd ../build
make
#link
cd ../allsky
./make_links.sh
#initialize
python3 allsky_init.py

#run the code
echo $1 > config.txt
python3 allsky_run.py
#compare the precision
python3 compare-to-reference.py