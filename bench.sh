#!/bin/bash


set -x
upcxx-run -v -n 96 -N 1 -shared-heap 20% -vv -- src/mhm2 -r /global/D1/projects/ipumer/meta_hip_mer/portal.nersc.gov/project/hipmer/MetaHipMer_datasets_12_2019/ArcticSynth/arcticsynth.fastq --progress -o arctic-dual