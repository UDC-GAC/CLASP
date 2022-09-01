
mkdir tmp

############################### ResNet-50 ###############################

mkdir tmp/rn50_bs

python -u bench.py  1 1 Sputnik rn50 > tmp/rn50_bs/Sputnik_half_rn50_k1_m16n8k8

python -u bench.py  2 2 CLASP rn50 > tmp/rn50_bs/CLASP_half_rn50_k2_v2_m16n8k8
python -u bench.py  4 4 CLASP rn50 > tmp/rn50_bs/CLASP_half_rn50_k4_v4_m16n8k8
python -u bench.py  8 8 CLASP rn50 > tmp/rn50_bs/CLASP_half_rn50_k8_v8_m16n8k8
python -u bench.py 16 8 CLASP rn50 > tmp/rn50_bs/CLASP_half_rn50_k16_v8_m16n8k8
python -u bench.py 32 8 CLASP rn50 > tmp/rn50_bs/CLASP_half_rn50_k32_v8_m16n8k8

############################### Transformer ###############################

mkdir tmp/transformer_bs

python -u bench.py  1 1 Sputnik transformer > tmp/transformer_bs/Sputnik_half_transformer_k1_m16n8k8

python -u bench.py  2 2 CLASP transformer > tmp/transformer_bs/CLASP_half_transformer_k2_v2_m16n8k8
python -u bench.py  4 4 CLASP transformer > tmp/transformer_bs/CLASP_half_transformer_k4_v4_m16n8k8
python -u bench.py  8 8 CLASP transformer > tmp/transformer_bs/CLASP_half_transformer_k8_v8_m16n8k8
python -u bench.py 16 8 CLASP transformer > tmp/transformer_bs/CLASP_half_transformer_k16_v8_m16n8k8
python -u bench.py 32 8 CLASP transformer > tmp/transformer_bs/CLASP_half_transformer_k32_v8_m16n8k8