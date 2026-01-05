"""
Load the instadeep version of the clinvar dataset, save it with a specific sequence length, and reformat it for use in this codebase
Note: this requires a version of datasets that allows scripts (datasets<3)
"""

import argparse
from datasets import Dataset, DatasetDict, load_dataset
from Bio import SeqIO
from tqdm import tqdm


def process_example(item):
    ref_seq = item["ref_forward_sequence"]
    alt_seq = item["alt_forward_sequence"]
    mid_point = len(ref_seq) // 2
    if ref_seq[mid_point] == alt_seq[mid_point]:
        mid_point -= 1
    ref_bp = ref_seq[mid_point]
    alt_bp = alt_seq[mid_point]
    assert ref_bp != alt_bp, (
        f"Got the same bp {ref_bp} for both ref and alt at position {mid_point}"
    )
    return {"seq": item["ref_forward_sequence"], "ref": ref_bp, "alt": alt_bp, "MAF": 0}


def main(args):
    dataset = load_dataset(
        "InstaDeepAI/genomics-long-range-benchmark",
        task_name="variant_effect_pathogenic_clinvar",
        sequence_length=args.seq_len,
    )
    print(dataset)
    new_ds = dataset.map(process_example)
    print(new_ds)
    new_ds.push_to_hub(f"emarro/clinvar_vep_{args.seq_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    args = parser.parse_args()
    main(args)
