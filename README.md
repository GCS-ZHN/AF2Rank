# AF2Rank
This version of af2rank is from [colabfold](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/AF2Rank.ipynb#scrollTo=UCUZxJdbBjZt) version.

对生成的序列进行评估

```bash
$af2rank seq --native-pdb $native_pdb \
            --chain A,B \
            --recycles 4 \
            --model-type alphafold-multimer \
            --mask-sequence \
            --mask-sidechains \
            --save-pdb \
            --seq-file $seq_file
```

对已有的pdb进行评估。
```bash
$af2rank structure --native-pdb $native_pdb \
            --chain A \
            --recycles 4 \
            --model-type alphafold \
            --mask-sequence \
            --mask-sidechains \
            --save-pdb \
            --pdb-dir $pdb_dir
```