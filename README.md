# ARTEM

**A**ligning **R**NA **TE**rtiary **M**otifs (ARTEM) is a tool for superimposing arbitrary RNA spatial structures

# Contents
<!-- - [ARTEM](#artem)
- [Contents](#contents) -->
- [How it works](#how-it-works)
  - [Coordinate representation of nucleotides](#coordinate-representation-of-nucleotides)
  - [Examples](#examples)
- [Installation](#installation)
- [Usage](#usage)
  - [Command](#command)
  - [Options](#options)
- [Requirements](#requirements)
- [Time & Memory usage](#time--memory-usage)
- [Contacts](#contacts)

# How it works

The ARTEM method of superimposing an arbitrary RNA spatial structure $Y$ on the RNA structure $X$ is performed in two main steps:

1. superimposition of $Y$ on $X$ by a residue pair $p=(X_i, Y_j)$  
2. superimposition of $Y$ on $X$ by the set of residue pairs $s=[(X_{i_0}, Y_{i_0}), (X_{i_1}, Y_{i_1}), ..., (X_{i_k}, Y_{i_k})]$, which are mutually nearest as a result of superimposition 1

In both cases, the superposition is an application of the linear operator $L$ to the structure $Y$, which minimizes RMSD on a given $p$ or $s$ set of residue pairs. Finding such an operator $L$ is implemented by the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) via SVD decomposition.
 
ARTEM tool performs steps 1-2 for all possible or user-restricted residue pairs between the $X$ and $Y$ structures. On the standard output is fed a table of matches between $p$ and $s$ sets of residue pairs in DSSR format and metrics SIZE of set $s$, RMSD of superposition and the value of RMSD / SIZE. On request, ARTEM saves all defined Y structure impositions in the selected format (.pdb or .cif)


## Coordinate representation of nucleotides

ARTEM uses four representations for each nucleotide residue of the $X$ and $Y$ structures:

1. the five-atomic representation for finding a primary alignment operator $L_p$  
2. center mass of the nucleotide, calculated using the five-atomic representation to find the mutually nearest nucleotides in step 2  
3. the three-atomic representation for finding the secondary alignment operator $L_s$  
4. the three-atomic representation for calculating the final RMSD metric  

> A residue that does not have at least one of these representations will be ignored by ARTEM in all 1-2 main calculation steps.

The five-atomic representation used in representations 1-2 includes a phosphate atom, a ribose center of mass, and three base atoms:

- for Purines  
<span style="color:#b00b13">P</span>, <span style="color:#6aa84f ">C1' C2' O2' C3' O3' C4' O4' C5' O5'</span>, <span style="color:#8e7cc3">N9</span>, <span style="color:#674ea7">C2</span>, <span style="color:#351c75">C6</span>
- for Pyrimidines  
<span style="color:#b00b13">P</span>, <span style="color:#6aa84f ">C1' C2' O2' C3' O3' C4' O4' C5' O5'</span>, <span style="color:#8e7cc3">N1</span>, <span style="color:#674ea7">C2</span>, <span style="color:#351c75">C4</span>
- for Pyrimidins with a C-glycosidic bond  
<span style="color:#b00b13">P</span>, <span style="color:#6aa84f ">C1' C2' O2' C3' O3' C4' O4' C5' O5'</span>, <span style="color:#8e7cc3">C5</span>, <span style="color:#674ea7">C4</span>, <span style="color:#351c75">C2</span>

The three-atomic representations used in the representations 3-4 include three base atoms:

- for Purines  
<span style="color:#8e7cc3">N9</span>, <span style="color:#674ea7">C2</span>, <span style="color:#351c75">C6</span>
- for Pyrimidines  
<span style="color:#8e7cc3">N1</span>, <span style="color:#674ea7">C2</span>, <span style="color:#351c75">C4</span>
- for Pyrimidins with a C-glycosidic bond  
<span style="color:#8e7cc3">C5</span>, <span style="color:#674ea7">C4</span>, <span style="color:#351c75">C2</span>


The five- and three-atomic representations of specific nucleotides can be learned from the nucleotide atomic representation library [lib/nar.py](lib/nar.py). Here new nucleotides can be manually added or representations for existing nucleotides can be changed, if necessary.

## **Examples**

# Installation
Clone the GitHub repository by typing

    git clone https://github.com/david-bogdan-r/ARTEM.git

# Usage
## Command

    python3 artem.py r=FILENAME q=FILENAME rres=STRING qres=STRING saveto=FOLDER saveres=STRING rmsdmin=FLOAT rmsdmax=FLOAT sizemin=FLOAT sizemax=FLOAT rmsdsizemin=FLOAT rmsdsizemax=FLOAT matchrange=FLOAT rformat=KEYWORD qformat=KEYWORD saveformat=KEYWORD threads=INT rseed=STRING qseed=STRING rresneg=STRING qresneg=STRING

## Options

    r=FILENAME
        Path to the reference structure in PDB/mmCIF format.

    q=FILENAME
        Path to the query structure (the one that ARTEM superimpose to  
        reference) in PDB/mmCIF format.

    rres=STRING, qres=STRING
        List of specifying the residues that ARTEM consider as part of  
        the reference/query structure (and ignore all the other residues).
        See res options format below.

    rresneg=STRING, qresneg=STRING
        The inversion of rres and qres - ARTEM ignore all the residues that  
        correspond to resneg parameters. If both rres and rresneg are specified  
        simultaneously - ARTEM ignore rres and use rresneg (i.e. negative  
        parameters have higher priority.
        See res options format below.

    rseed=STRING, qseed=STRING
        The subsets of rres and qres that will be used as seeds.  
        The nomenclature used is similar to the res parameters, see it below.

    saveto=FOLDER
        Path to the folder where impositions of the queried structure will  
        be saved. If the folder does not exist, it will be created.

    saveres=STRING
        List of specifying the residues of the queried structure ARTEM  
        will save. By default the value of qres is used.
        See res options format below.

    rmsdmin=FLOAT, rmsdmax=FLOAT, sizemin=FLOAT, sizemax=FLOAT,  
    rmsdsizemin=FLOAT, rmsdsizemax=FLOAT, matchrange=FLOAT  
        Thresholds for RMSD (rmsdmin, rmsdmax), SIZE (sizemin, sizemax),  
        RMSD/SIZE (rmsdsizemin, rmsdsizemax), and the maximal range for  
        mutually closest residues (matchrange). All the min/max parameters  
        (except matchrange) define which superpositions will be printed to  
        stdout (and saved to the saveto folder).  
        The defaults:
            rmsdmin     = 0.
            rmsdmax     = 1e10
            sizemin     = 1.
            sizemax     = 1e10
            rmsdsizemin = 0.
            rmsdsizemax = 1e10
            matchrange  = 3.

    rformat=KEYWORD, qformat=KEYWORD, saveformat=KEYWORD
        Formats of the input reference structure (rformat), query structure  
        (qformat), and the output superpositions to be saved (saveformat).  
        By default rformat and qformat are set to PDB, saveformat by default  
        is set to be equal to qformat. When specified and equal to "pdb"  
        (case-insensitive) - ARTEM set it to "PDB", when equal to "cif" or  
        "mmcif" (case-insensitive) - ARTEM set it to "CIF", otherwise raise  
        an error.

    threads=INT
        Number of CPUs to use. 1 by default. ARTEM multiprocessing is  
        available only for UNIX-like systems.

    ***
    res options format

    [#[INT]][/[STRING]][:[STRING][_INT[CHAR|_INT]]
    
        #[INT]
            Reference/query structure model number. If not specified, the 
             default is the first model #1. If an empty # number is specified,  
             all models of the structure are used.
        /[STRING]
            Chain label
        :[STRING][_INT[CHAR|_INT]
            Range and/or types of residues
            :STRING
                to determine the residues with a specified type (e.g. ':A', ':G' etc.)
            :_INT[CHAR]
                to determine the residue with the specified index INT and  
                the specified type CHAR
            :_INT_INT
                to determine the residues with indexes in a given range



# Requirements
numpy==1.22.3  
pandas==1.4.2  
scipy==1.8.1  

# Time & Memory usage
The implementation with default parameters takes around one minute to run an entire 5,970-residue eukaryotic ribosome (PDB entry 7O7Y) against a 160-residue TPP riboswitch (PDB entry 2GDI) on 32 cores, taking under 2Gb RAM at peak on an AMD Ryzen 9 5950X machine with 128Gb RAM. On the same machine on 32 cores a run of a 2,828-residue LSU rRNA (PDB entry 1FFK) against itself requires 20 minutes in time and 70Gb of RAM.

# Contacts

David Bogdan  
*e-mail: bogdan.d@phystech.edu*  
*telegram: https://t.me/david_bogdan*