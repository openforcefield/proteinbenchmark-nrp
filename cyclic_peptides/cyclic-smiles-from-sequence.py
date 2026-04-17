#! /usr/bin/env python3

olc_to_smiles = {
    "G": "N[CH2]C(=O)",
    "V": "N[C@H](C(C)C)C(=O)",
    "F": "N[C@H](Cc2ccccc2)C(=O)",
    "A": "N[C@H](C)C(=O)",
    "T": "N[C@H](C(O)C)C(=O)",
    "S": "N[C@H](CO)C(=O)",
    "R": "N[C@H](CCCNC(=[NH2+])[NH2])C(=O)",
    "N": "N[C@H](CC(=O)[NH2])C(=O)",
    "D": "N[C@H](CC(=O)[O-])C(=O)",
    "v": "N[C@@H](C(C)C)C(=O)",
    "s": "N[C@@H](CO)C(=O)",
    "a": "N[C@@H](C)C(=O)",
}

def main(sequence: str, cyclic: bool = True):
    smiles = "".join([olc_to_smiles[c] for c in sequence])
    if cyclic:
        assert smiles[0] == "N"
        assert "1" not in smiles
        smiles = smiles[0] + "1" + smiles[1:]
        assert smiles[-5:] == "C(=O)"
        smiles = smiles[:-4] + "1" + smiles[-4:]
    return smiles


if __name__ == "__main__":
    import cyclopts

    cyclopts.run(main)
