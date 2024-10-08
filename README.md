# ARTEM

**A**ligning **R**NA **TE**rtiary **M**otifs (ARTEM) is a tool for superimposing arbitrary RNA spatial structures

## References

[E.F. Baulin, D.R. Bohdan, D. Kowalski, M. Serwatka, J. Świerczyńska, Z. Żyra, J.M. Bujnicki, (2024) ARTEM: a method for RNA tertiary motif identification with backbone permutations, and its example application to kink-turn-like motifs. bioRxiv. DOI: 10.1101/2024.05.31.596898](https://doi.org/10.1101/2024.05.31.596898)

[D.R. Bohdan, V.V. Voronina, J.M. Bujnicki, E.F. Baulin (2023) A comprehensive survey of long-range tertiary interactions and motifs in non-coding RNA structures. Nucleic Acids Research. gkad605. DOI: 10.1093/nar/gkad605](https://doi.org/10.1093/nar/gkad605)

## Check out [our other developments](https://github.com/febos/wiki)

## How ARTEM works

ARTEM reads a reference and a query structure from the specified coordinate files in PDB or in mmCIF format, and, by default, prints a sorted list of their local structural superpositions. The user can choose to save the superimposed versions of the query structure into a specified folder in PDB or in mmCIF format. Each of the saved files will include three models: (1) the entire (according to the "qres" parameter) superimposed query structure, (2) the subset of the reference structure residues used for the superposition, and (3) the subset of the query structure residues used for the superposition. By default, ARTEM reads the entire first models of the input files.

The ARTEM algorithm works as follows. For each possible single-residue matching seed between the reference and the query structures (as defined by "rseed" and "qseed" parameters) ARTEM superimposes the structures based on the 5-atom representations of the two residues. Then, ARTEM searches for a subset of the reference and query residues that are mutually closest to each other in the current superposition. Finally, ARTEM performs a second superposition using the subset of the mutually closest residues as the assigned residue-residue matchings. Finally, ARTEM prints a sorted list of the produced superpositions to stdout. For each superposition the output includes its ID, RMSD, SIZE, RMSD/SIZE ratio, the list of generative single-residue seeds (PRIM), and the list of the residue-residue matchings (SCND).

## Installation

Clone the GitHub repository by typing

    git clone https://github.com/david-bogdan-r/ARTEM.git

## Dependencies

ARTEM requires four Python3 libraries to be installed:

- numpy
- pandas
- requests
- scipy

To install, type:

    pip install -r requirements.txt

ARTEM was tested with two different Python3 environments:

### Ubuntu 20.04

- python==3.8
- numpy==1.22.3
- pandas==1.4.1
- requests==2.31.0
- scipy==1.8.0

### MacOS Sonoma 14.5

- python==3.12
- numpy==1.26.3
- pandas==2.1.4
- requests==2.31.0
- scipy==1.11.4

## Time & memory usage

The implementation with default parameters takes around one minute to run an entire 5,970-residue eukaryotic ribosome (PDB entry 7O7Y) against a 160-residue TPP riboswitch (PDB entry 2GDI) on 32 cores, taking under 2Gb RAM at peak on an AMD Ryzen 9 5950X machine with 128Gb RAM. On the same machine on 32 cores a run of a 2,828-residue LSU rRNA (PDB entry 1FFK) against itself requires 20 minutes in time and 70Gb of RAM.

## Usage

    python3 artem.py r=FILENAME q=FILENAME [OPTIONS]

## Usage examples

    1) python3 artem.py r=examples/1ivs.cif  q=examples/1wz2.cif rres=/C qres=/C > output.txt

    This command will write into "output.txt" file a sorted list of all local 
    structural superpositions between the C chains of 1IVS and 1WZ2 PDB entries 
    that are two tRNAs. The user can spot the three largest mathings of size 52. 
    Then the user can save the largest mathings only into "result" folder in 
    PDB format:

    python3 artem.py r=examples/1ivs.cif  q=examples/1wz2.cif rres=/C qres=/C sizemin=52 saveto=result saveformat=pdb

    As the result three files of three different matchings of 52 residues will 
    be saved in PDB format - 1wz2_1.pdb, 1wz2_2.pdb, 1wz2_3.pdb. The latter is
    the superposition with the best RMSD. Each of the saved files lists three 
    structural models. The first model contains all "qres" residues 
    superimposed with the reference structure. The second model contains 
    the subset of the reference structure residues that were used 
    for the superposition, and the third model stores their counterpart 
    residues from the query structure. Then, the user can open the reference 
    file 1ivs.cif together with the first model of the file 1wz2_3.pdb in a 3D
    visualization tool for visual examination of the best local superposition 
    of the two structures. The user can observe a good superposition 
    of the four standard tRNA helical regions.

    2) python3 artem.py r=examples/motif10.pdb  q=examples/motif9.pdb rmsdsizemax=0.25

    This command will output a sorted list of those local structural 
    superpositions between the two topologically different motifs of 
    10 and 9 residues respectively that have a ration RMSD/SIZE under 0.25.
    The user can spot the only mathing of size 8 with RMSD of 1.713 angstroms.
    Then the user can save the superposition into "result" folder 
    in CIF format:

    python3 artem.py r=examples/motif10.pdb  q=examples/motif9.pdb rmsdsizemax=0.25 sizemin=8 saveto=result saveformat=cif 

    The only file will be saved named "motif9_1.cif". Then, the user 
    can open the reference file motif10.pdb together with the file 
    motif9_1.cif in a 3D visualization tool for visual examination. 
    The user can observe a good superposition of all the three stacked 
    base pairs. Simultaneously, two of the three A-minor-forming 
    adenosines have a counterpart residue.

    3) python3 artem.py r=examples/test_data q=examples/motif10.pdb sizemin=9 saveto=result

    Motif search example 1.
    Report the matches of at least 9 residues between the query structure motif10.pdb and
    all the pdb files in the examples/test_data folder. 
    Save the results under the result folder.
    
    4) python3 artem.py r=examples/test_data q=examples/motif10.pdb sizemin=9 qrst="/R:_151_153 |R| /R:_151_153 |strand|"

    Motif search example 2.
    Report the matches between the query structure motif10.pdb and
    all the pdb files in the examples/test_data folder, 
    where the query residues 151, 152, and 153 of chain R 
    match with a continuous strand of purines.

## Options

    r=FILENAME/FOLDER/PDB-ENTRY [REQUIRED OPTION]
        Path to a reference structure in PDB/mmCIF format. For faster 
        performance, it's advised to specify the largest of the two structures 
        as the reference structure.
        If a folder or a mask is specified instead, ARTEM will process 
        all the PDB/mmCIF files (according to the rformat parameter) 
        in that folder/mask as a reference structure one by one.
        If a 4-character PDB entry is specified, ARTEM will download the
        structure from RCSB PDB.

    q=FILENAME/FOLDER/PDB-ENTRY [REQUIRED OPTION]
        Path to a query structure, the one that ARTEM superimposes to 
        the reference, in PDB/mmCIF format.
        If a folder or a mask is specified instead, ARTEM will process 
        all the PDB/mmCIF files (according to the qformat parameter) 
        in that folder/mask as a query structure one by one.
        If a 4-character PDB entry is specified, ARTEM will download the
        structure from RCSB PDB.

    matchrange=FLOAT [DEFAULT: matchrange=3.0]
        The threshold used for searching the mutually closest residues. Only 
        those pairs of residues that have their centers of mass at a distance 
        under the specified matchrange value can be added to the subset 
        of the mutually closest residues. The higher matchrange value 
        will produce more "noisy" matchings but won't miss anything. The lower 
        matchrange value will produce more "clean" matchings but 
        can miss something.

    rformat=KEYWORD, qformat=KEYWORD [DEFAULT: rformat=PDB,qformat=PDB] 
        The specification of the input coordinate file formats 
        (case-insensitive). By default, ARTEM tries to infer the format 
        from the extensions of the input filenames. ".pdb", ".cif", 
        and ".mmcif" formats can be recognized (case-insensitive). In the case 
        of any other extension ARTEM will treat the file as the PDB-format 
        file by default. If the "rformat" ("qformat") parameter is specified 
        and it's either "PDB", "CIF", or "MMCIF" (case-insensitive), 
        ARTEM will treat the reference (query) coordinate file
        as the specified format.

    rmsdmin=FLOAT, rmsdmax=FLOAT [DEFAULT: rmsdmin=0,rmsdmax=1e10]
        The specification of minimum and maximum RMSD thresholds. 
        ARTEM will print and save only the superpositions that satisfy 
        the specified thresholds. 

    rmsdsizemin=FLOAT, rmsdsizemax=FLOAT [DEFAULT: rmsdsizemin=0,rmsdsizemax=1e10]
        The specification of minimum and maximum RMSD/SIZE ratio thresholds. 
        ARTEM will print and save only the superpositions that satisfy 
        the specified thresholds. 

    resrmsdmin=FLOAT, resrmsdmax=FLOAT [DEFAULT: resrmsdmin=0, resrmsdmax=1e10]
        The specification of minimum and maximum per-residue RMSD threshold.
        ARTEM will print and save only the superpositions that satisfy 
        the specified thresholds.

    rnosub=BOOL, qnosub=BOOL [DEFAULT: rnosub=False, qnosub=False]
        Omit the sub-matches. If rnosub=True, ARTEM will not output the matches
        representing reference residue subsets of any other match, i.e. only
        the largest matches will be reported. Analogously if qnosub=True,
        ARTEM will not report the matches that are query residue subsets of any
        other match.

    rres=STRING, qres=STRING [DEFAULT: rres="#1",qres="#1"]
        The specification of the input reference (rres) and query (qres) 
        structures. Only the specified residues will considered as part 
        of the structure and all the other residues will be ignored. 
        See the format description at the end of the OPTIONS section.

    rresneg=STRING, qresneg=STRING [DEFAULT: None]
        The specification of the input reference (rresneg) and query (qresneg) 
        structures. The specified residues will be ignored and all the other 
        residues considered as part of the structure.
        See the format description at the end of the OPTIONS section.

    rrst=STRING, qrst=STRING [DEFAULT: None]
        The specification of the input reference (rrst) and query (qrst) 
        structure restraints for post-filtering of the matches. 
        The format: 
        "ChimeraX-like spec | restraint | ChimeraX-like spec | restraint |"
        See the description of the ChimeraX-like format at the end 
        of the OPTIONS section.
        The possible restraints:
            || - (empty restraint) report only the matches where the specified
                residues have the matched counterparts.
            |1.5| - report only the matches where the specified residues 
                have the matched counterparts and their total RMSD 
                is under 1.5 angstroms.
            |A| - report only the matches where the specified residues matched
                with adenosines. IUPAC nomenclature is allowed.
            |strand| - report only the matches where the specified residues 
                matched with a continuous strand (defined as all O3'-P pairs under 
                2.0 angstroms).
        Examples:
            rrst = "/A:_10_20"  [or equiv. "/A:_10_20 ||"] 
                Report only the matches where the reference structure residues
                in chain A with identifiers from 10 to 20 are all among the 
                matched residues.
            qrst = " /A:DG_10 |DG| /B:G_2 /B:C_3 |strand| /B:G_2 /B:C_3 |1.0|" 
                Report only the matches where the three query residues /A:DG_10,
                /B:G_2, /B:C_3 are present among the matched residues, 
                the /A:DG_10 is matching a DG residue in the reference structure,
                and the /B:G_2 and /B:C_3 match a pair of consecutive residues
                under 1.0 angstrom RMSD.

    rseed=STRING, qseed=STRING [DEFAULT: rseed=rres,qseed=qres]
        The specification of the reference and query residues that ARTEM can use
        for single-residue matching seeds.
        See the format description at the end of the OPTIONS section.

    saveformat=KEYWORD [DEFAULT: saveformat=qformat]
        The specification of the format of the output coordinate files. 
        By default, ARTEM will save the coordinate files in the same format 
        as the query input file. If the "saveformat" parameter is specified 
        and it's either "PDB", "CIF", or "MMCIF" (case-insensitive), ARTEM 
        will save the output coordinate files in the specified format.

    saveres=STRING [DEFAULT: saveres=qres]
        The specification of the query structure residues that will be saved 
        in the output coordinate files.
        See the format description at the end of the OPTIONS section.

    saveto=FOLDER [DEFAULT: None]
        Path to the output folder to save the coordinate files 
        of the superimposed query structures along with the mutually 
        closest residue subsets. If the specified folder does not exist, 
        ARTEM will create it. In the case of the reference structure input 
        being a folder (r=FOLDER), ARTEM will create a sub-folder inside 
        the saveto folder for each of the input reference structures. 
        If the saveto folder is not specified, nothing will be saved. 

    sizemin=FLOAT, sizemax=FLOAT [DEFAULT: sizemin=1,sizemax=1e10]
        The specification of minimum and maximum SIZE thresholds. 
        ARTEM will print and save only the superpositions that satisfy 
        the specified thresholds. If sizemin is set to zero, ARTEM will 
        output the dummy 0-size superpositions matching the reference 
        and query residues lacking the 5-atom representation specified. 
        The user can specify custom atom representations of any residues 
        via editing the lib/nar.py text file.

    trim=BOOL [DEFAULT: trim=None]
        When specified, for each subset of mutually closest residues ARTEM will 
        iteratively remove residues with the worst per-residue RMSD from the 
        subset one by one with the following re-superpositioning based on the 
        remaining residues of the subset until the specified thresholds for rmsdmax,
        rmsdsizemax, resrmsdmax are reached or the subset size is less than sizemin.

    threads=INT [DEFAULT: threads= CPU COUNT]
        Number of CPUs to use. ARTEM multiprocessing is available only for 
        UNIX-like systems.

    silent=BOOL [DEFAULT: silent=False]
        If specified, ARTEM will not raise any errors.

    ***********************************************************************
    ARTEM uses a ChimeraX-like format to specify the residues of interest 
    using the "res" parameters:

    [#[INT]][/[STRING]][:[STRING][_INT[CHAR|_INT]] - The structure specification
                                                     format. The "res" 
                                                     parameters can be defined 
                                                     with a number 
                                                     of specifications 
                                                     separated by spaces and 
                                                     enclosed in double quotes.

        #[INT]                    == Model number
        /[STRING]                 == Chain identifier
        :[STRING][_INT[CHAR|_INT] == Residue(s) specification:

            :STRING               == Residue type    
            :_INT[CHAR]           == Residue number [with insertion code]
            :_INT_INT             == Range of residue numbers

    Structure specification examples:

        rres="#1/B:_-10_20 #1/A"    - Consider the entire chain A from model 1 
                                      and the range of chain B residues with 
                                      numbers from -10 to 20 from model 1 as 
                                      the reference structure.
        qres="#"                    - Consider all the residues from all 
                                      the models in the "q" file as 
                                      the query structure.
        saveres="/C:_10_20 /C:_20A" - Save the chain C residues with numbers 
                                      from 10 to 20 and the chain C residue 
                                      with number 20A (A is the insertion code).
        rseed=:A                    - Use only the model 1 adenosines as the 
                                      single-residue seeds from the reference 
                                      structure.

## Contacts

David Bogdan, *e-mail: <bogdan.d@phystech.edu>*
