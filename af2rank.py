#!/work/home/zhanghongning/software/af2rank/1.0/.conda/bin/python3
import os
import numpy as np
import warnings
import pandas as pd
import tempfile
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign import clear_mem, mk_af_model
from colabdesign.shared.utils import copy_dict
from protools.protools.seqio import read_fasta


def save_temp_pdb(x, prefix="temp"):
    temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".pdb")
    with open(temp.name,"w") as out:
        for k,c in enumerate(x):
            out.write(
                "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                 % (k+1,"CA","ALA","A",k+1,c[0],c[1],c[2],1,0))
    return temp


def tmscore(x,y):
    # save to dumpy pdb files
    temp1 = save_temp_pdb(x)
    temp2 = save_temp_pdb(y)
    # pass to TMscore
    output = os.popen(f'TMscore {temp1.name} {temp2.name}')

    # parse outputs
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    o = {}
    for line in output:
        line = line.rstrip()
        if line.startswith("RMSD"): o["rms"] = parse_float(line)
        if line.startswith("TM-score"): o["tms"] = parse_float(line)
        if line.startswith("GDT-TS-score"): o["gdt"] = parse_float(line)

    return o


class af2rank:
    def __init__(
            self, 
            pdb, 
            chain=None, 
            model_name="model_1_ptm", 
            model_names=None,
            model_dir='.'):
        self.args = {"pdb":pdb, "chain":chain,
                                 "use_multimer":("multimer" in model_name),
                                 "model_name":model_name,
                                 "model_names":model_names,
                                 "model_dir": model_dir}
        self.reset()

    def reset(self):
        self.model = mk_af_model(protocol="fixbb",
                                                         use_templates=True,
                                                         use_multimer=self.args["use_multimer"],
                                                         debug=False,
                                                         model_names=self.args["model_names"],
                                                         data_dir=self.args["model_dir"])

        self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
        self.model.set_seq(mode="wildtype")
        self.wt_batch = copy_dict(self.model._inputs["batch"])
        self.wt = self.model._wt_aatype

    def set_pdb(self, pdb, chain=None):
        if chain is None: chain = self.args["chain"]
        self.model.prep_inputs(pdb, chain=chain)
        self.model.set_seq(mode="wildtype")
        self.wt = self.model._wt_aatype

    def set_seq(self, seq):
        self.model.set_seq(seq=seq)
        self.wt = self.model._params["seq"][0].argmax(-1)

    def _get_score(self):
        score = copy_dict(self.model.aux["log"])

        score["plddt"] = score["plddt"]
        score["pae"] = 31.0 * score["pae"]
        score["rmsd_io"] = score.pop("rmsd",None)

        i_xyz = self.model._inputs["batch"]["all_atom_positions"][:,1]
        o_xyz = np.array(self.model.aux["atom_positions"][:,1])

        # TMscore to input/output
        if hasattr(self,"wt_batch"):
            n_xyz = self.wt_batch["all_atom_positions"][:,1]
            score["tm_i"] = tmscore(n_xyz,i_xyz)["tms"]
            score["tm_o"] = tmscore(n_xyz,o_xyz)["tms"]

        # TMscore between input and output
        score["tm_io"] = tmscore(i_xyz,o_xyz)["tms"]

        # composite score
        score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
        return score

    def predict(self, pdb=None, seq=None, chain=None,
                            input_template=True, model_name=None,
                            rm_seq=True, rm_sc=True, rm_ic=False,
                            recycles=1, iterations=1,
                            output_pdb=None, extras=None, verbose=True):

        if model_name is not None:
            self.args["model_name"] = model_name
            if "multimer" in model_name:
                if not self.args["use_multimer"]:
                    self.args["use_multimer"] = True
                    self.reset()
            else:
                if self.args["use_multimer"]:
                    self.args["use_multimer"] = False
                    self.reset()

        if pdb is not None: self.set_pdb(pdb, chain)
        if seq is not None: self.set_seq(seq)

        # set template sequence
        self.model._inputs["batch"]["aatype"] = self.wt

        # set other options
        self.model.set_opt(
                template=dict(rm_ic=rm_ic),
                num_recycles=recycles)
        self.model._inputs["rm_template"][:] = not input_template
        self.model._inputs["rm_template_sc"][:] = rm_sc
        self.model._inputs["rm_template_seq"][:] = rm_seq

        # "manual" recycles using templates
        ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
        for i in range(iterations):
            self.model.predict(models=self.args["model_name"], verbose=False)
            if i < iterations - 1:
                self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
            else:
                self.model._inputs["batch"]["all_atom_positions"] = ini_atoms

        score = self._get_score()
        if extras is not None:
            score.update(extras)

        if output_pdb is not None:
            self.model.save_pdb(output_pdb)

        if verbose:
            print_list = ["tm_i","tm_o","tm_io","composite","ptm","i_ptm","plddt","fitness","id"]
            print_score = lambda k: f"{k} {score[k]:.4f}" if isinstance(score[k],float) else f"{k} {score[k]}"
            print(*[print_score(k) for k in print_list if k in score])

        return score
    

def rank_structure(
        decoys_dir: str,
        af: af2rank,
        save_pdb: bool = True,
        **kwargs) -> pd.DataFrame:
        scores = []
        for decoy_pdb in os.listdir(decoys_dir):
            input_pdb = os.path.join(decoys_dir, decoy_pdb)
            if save_pdb:
                os.makedirs(f"{decoys_dir}_output", exist_ok=True)
                output_pdb = os.path.join(f"{decoys_dir}_output",decoy_pdb)
            else:
                output_pdb = None
            scores.append(af.predict(pdb=input_pdb, output_pdb=output_pdb,
                                                            **kwargs, extras={"id":decoy_pdb}))
        return pd.DataFrame(scores)


def rank_seq(seqs: list, af: af2rank, output_dir: str = None, **kwargs) -> pd.DataFrame:
    scores = []
    for sid, seq in seqs:
        output_pdb = None
        if output_dir:
            output_pdb = os.path.join(output_dir, f'{sid}.pdb')
        score = af.predict(
            seq=seq,
            output_pdb=output_pdb,
            extras={'id':sid},
            **kwargs
        )
        scores.append(score)
    return pd.DataFrame(scores)


def main():
    from argparse import ArgumentParser
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument('--native-pdb', required=True)
    common_parser.add_argument('--chain', required=True)
    common_parser.add_argument('--recycles', default=1, type=int)
    common_parser.add_argument('--iterations', default=1, type=int)
    common_parser.add_argument(
        '--model-type', 
        default='alphafold', 
        choices=['alphafold', 'alphafold-multimer'])
    common_parser.add_argument(
        '--model-num',
        default=1,
        type=int
    )
    common_parser.add_argument('--mask-sequence', action='store_true')
    common_parser.add_argument('--mask-sidechains', action='store_true')
    common_parser.add_argument('--mask-interchain', action='store_true')
    common_parser.add_argument('--save-pdb', action='store_true')

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    seq_rank_parser = subparsers.add_parser('seq', parents=[common_parser])
    seq_rank_parser.add_argument('--seq-file', required=True)

    strc_rank_parser = subparsers.add_parser('structure', parents=[common_parser])
    strc_rank_parser.add_argument('--pdb-dir', required=True)

    args = parser.parse_args()

    if args.model_type == "alphafold":
        model_name = f"model_{args.model_num}_ptm"
    if args.model_type == "alphafold-multimer":
        model_name = f"model_{args.model_num}_multimer_v3"

    SETTINGS = {"rm_seq":args.mask_sequence,
                "rm_sc":args.mask_sidechains,
                "rm_ic":args.mask_interchain,
                "recycles":args.recycles,
                "iterations":args.iterations,
                "model_name":model_name}
    
    clear_mem()
    af = af2rank(
        pdb=args.native_pdb,
        chain=args.chain,
        model_name=model_name,
        model_dir=f'{os.path.dirname(__file__)}/params')
    
    if args.subcommand == 'structure':
        res = rank_structure(args.pdb_dir, af, args.save_pdb, **SETTINGS)
        res.to_csv(
            f'{args.pdb_dir}_af2rank.csv'
        )

    elif args.subcommand == 'seq':
        if args.seq_file.endswith('.fasta') or args.seq_file.endswith('.fa'):
            seqs = read_fasta(args.seq_file)
            seqs_iter = map(lambda x: (x, str(seqs[x].seq)), seqs)
        elif args.seq_file.endswith('.csv'):
            seqs = pd.read_csv(args.seq_file)
            seqs_iter = map(lambda r: (r[1]['id'], r[1]['seq']), seqs.iterrows())
        else:
            raise RuntimeError("Unsupported file type.")
        
        out_dir = None
        if args.save_pdb:
            out_dir = args.seq_file + '_pdbs'
            os.makedirs(out_dir, exist_ok=True)
        res = rank_seq(
            seqs_iter,
            af,
            output_dir=out_dir,
            **SETTINGS
        )
        res.to_csv(
            f'{args.seq_file}_af2rank.csv'
        )
    
    else:
        raise ValueError(f'Unsupport commmand {args.subcommand}')


if __name__ == '__main__':
    main()
