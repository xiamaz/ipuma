#!/usr/bin/env python

from Bio.Seq import Seq
import sys

def find_kmer_in_line(lines, i, kmer, is_rc):
    compl = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    line = lines[i].strip()
    kmer_index = line.find(kmer)
    if kmer_index == -1 or kmer_index == 0 or kmer_index + len(kmer) == len(line):
        return False
    quals = lines[i + 2].strip()
    left_ext = line[kmer_index - 1]
    right_ext = line[kmer_index + len(kmer)]
    left_qual = 0 if ord(quals[kmer_index - 1]) - 33 < 20 else 1
    right_qual = 0 if ord(quals[kmer_index + len(kmer)]) - 33 < 20 else 1
    if is_rc:
        left_ext, right_ext = right_ext, left_ext
        left_qual, right_qual = right_qual, left_qual
        if left_ext != '_':
            left_ext = compl[left_ext]
        if right_ext != '_':
            right_ext = compl[right_ext]
    print(left_ext, left_qual, right_ext, right_qual, end=' ')
    print(line[0:kmer_index], '\x1B[91m', line[kmer_index:kmer_index + len(kmer)], '\x1B[0m', line[kmer_index + len(kmer):], sep='')
    return True
    
kmer = sys.argv[1]
kmer_rc = str(Seq(kmer).reverse_complement())

#print(kmer)
#print(kmer_rc)

num_found = 0
lines = sys.stdin.readlines()
for i in range(len(lines)):
    if find_kmer_in_line(lines, i, kmer, False):
        num_found += 1
    if find_kmer_in_line(lines, i, kmer_rc, True):
        num_found += 1
print('Found', num_found)

