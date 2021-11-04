Run `induce_alignments_NMF.py` to induce new alignments between verses in a source and target edition of bible in different languages.

Input argumants:

```
--save_path 
path to the folder where intersection, gdfa and predicted alignments betwen the verses in the source and target editions will be saved.
default: ../graph-align/NMF/predicted_alignments

--gold_file
path to the file that contains the gold alignments between source and target editions.
default: ../graph-align/data/gold-standards/helfi/helfi-fin-heb-gold-alignments_test.txt

--verse_alignments_path
path to the folder that contains initial intersection and gdfa alignments for all edition pairs that are in the `editions_file`
default: ../graph-align/data/initial_verse_alignments

--source_edition
name of the source edition, must be included in the `editions_file`
default: fin-x-bible-helfi

--target_edition
name of the target edition, must be included in the `editions_file`
default: heb-x-bible-helfi

--editions_file
path to the file that contains the languages and editions that will be used for inferring new alignment edges between source and target editions. Must include the source and target editions.
default: ../graph-align/data/edition_lists/helfi_edition_list.txt

--core_count
number of cores to be used for parallel processing.
default: 80

--seed
random seed
default: 42
```

This script saves three files to `save_path` that contain (i) initial intersection alignments, (ii) initial gdfa alignments, (iii) induced alignments netween source and target editions for all the verses in the gold alignments.

Run `evaluate_induced_alignments.py` to evaluate the induced alignments.

Input argumants:

```
--save_path
path to the folder to save the evaluation results.
default: ../graph-align/NMF/results

--gold_file
path to the file that contains the gold alignments between source and target editions.
default: ../graph-align/data/gold-standards/helfi/helfi-fin-heb-gold-alignments_test.txt

--predicted_alignments_file
path to the file that contains induced alignments between source and target editions.
default: ../graph-align/NMF/predicted_alignments/predicted_alignments_from_fin-helfi_to_heb-helfi_with_max_84_editions_for_2230_verses_NMF.txt

--intersection_alignments_file
path to the file that contains induced alignments between source and target editions.
default: ../graph-align/NMF/predicted_alignments/intersection_alignments_from_fin-helfi_to_heb-helfi_for_2230_verses.txt

--gdfa_alignments_file
path to the file that contains induced alignments between source and target editions.
default: ../graph-align/NMF/predicted_alignments/gdfa_alignments_from_fin-helfi_to_heb-helfi_for_2230_verses.txt

--source_edition
name of the source edition, must be included in the `editions_file`
default: fin-x-bible-helfi

--target_edition
name of the target edition, must be included in the `editions_file`
default: heb-x-bible-helfi
```