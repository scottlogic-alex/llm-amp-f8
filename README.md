# Mixed-precision fp8 LLM training

Haven't done the mixed-precision fp8 bit yet.  

The plan:

- [x] train Pythia with Dao flash-attn
- [ ] measure FLOPs
- [ ] add DeepSpeed fp8
- [ ] measure FLOPs in fp8

The thing I'm hoping to see is a big speedup on 4090.

## Setup

Install (most of the) dependencies:

```bash
conda create -n llm-fp8 python=3.11
conda activate llm-fp8
MAX_JOBS=2 pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

Build-from-source flash-attn layers that are missing from distribution:

```bash
conda activate llm-fp8
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/csrc/layer_norm
MAX_JOBS=4 python setup.py install
cd ../fused_dense_lib
MAX_JOBS=4 python setup.py install
```

## Run

```bash
python -m script.train
```