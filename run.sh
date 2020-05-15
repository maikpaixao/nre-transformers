
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python scripts/cnn.py"
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python scripts/cnn_pos.py"
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python scripts/cnn_path.py"
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python scripts/cnn_chunk.py"
oarsub -p "gpu >= 2" -l "/host=1/core=8/gpu=2", walltime=15:00:00 "python scripts/cnn_semantics.py"


