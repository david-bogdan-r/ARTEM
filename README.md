# ARTEM

**A**ligning **R**NA **TE**rtiary **M**otifs (ARTEM) is a tool for superimposing arbitrary spatial RNA structures

# Contents
- [How it works](#how-it-works)
  - [Requirements](#requirements)
  - [Time & Memory usage](#time--memory-usage)
- [Installation](#installation)
- [Usage](#usage)
- [Options](#options)
- [Output format](#output-format)
- [Examples](#examples)


# How it works

For two RNA structures $X$ and $Y$ of sizes $N$ and $M$, for every $(i, j)$ pair, where $1 ≤ i ≤ N$ and $1 ≤ j ≤ M$, do the following:
1. Superimpose two structures using the Kabsch algorithm [KABSCH-LINK] considering only 5-atom representations of the residues $X_i$ and $Y_j$;
2. Calculate the centers of mass of 5-atom representations of all the residues of the two superimposed structures;
3. Based on the calculated centers of mass identify a subset of mutually closest residues at a distance < MATCHRANGE Å, such that if a residue Xs is the closest counterpart to a residue Yt, the residue Yt is the closest counterpart to the residue Xs, and the distance between their centers of mass is less than MATCHRANGE Å, then the pair (Xs, Yt) belongs to the subset. 
4. Superimpose the structures using the Kabsch algorithm considering only 3-atom base representations of the mutually closest residues subset. Calculate RMSD on the set of atoms used for the superposition. Output RMSD and SIZE (mutually closest subset size) values, as well as the coordinates of the superimposed structure $Y$. 



see [lib.nar](lib/nar.py) for details


The performed structure superposition operations are based on the Kabsch algorithm.





## Requirements
## Time & Memory usage

# Installation

    git clone https://github.com/david-bogdan-r/ARTEM.git

# Usage

    python3 artem.py [options]

# Options

    r


    q
    rres
    qres
    rresneg
    qresneg
    rseed
    qseed

# Output format

# Examples
