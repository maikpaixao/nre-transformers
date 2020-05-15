
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python3 scripts/cnn.py --position"


