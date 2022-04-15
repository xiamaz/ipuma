#!/bin/bash

for d in exp*ipu2*/mhm2.log ; do
	perl /home/lukb/git/isc21/ipuma/src/mhm2_parse_run_log.pl "$d" 2>/dev/null | awk 'NR==2'
done
