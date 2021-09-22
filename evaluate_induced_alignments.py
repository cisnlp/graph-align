from utils import LOG, setup_dict_entry, calc_and_update_score, load_gold, get_verse_alignments
import argparse
import os

def main(args):
    pros, surs = load_gold(args.gold_file)

    save_name = os.path.basename(args.predicted_alignments_file)
    save_name = save_name[:-4] + "_results.txt"

    result_names = [
        "base_intersection",
        "base_gdfa",
        "NMF",
        "NMF + intersection",
        "NMF + gdfa"
    ]

    results_all = {}
    for name in result_names:
        setup_dict_entry(results_all, name, {"p_hit_count": 0, "s_hit_count": 0, "total_hit_count": 0, "gold_s_hit_count": 0, "prec": 0, "rec": 0, "f1": 0, "aer": 0})

    with open(args.predicted_alignments_file, 'r') as f_pred, \
        open(args.intersection_alignments_file, 'r') as f_inter, \
        open(args.gdfa_alignments_file, 'r') as f_gdfa:
        
        lines_predicted = f_pred.read().splitlines()
        lines_intersection = f_inter.read().splitlines()
        lines_gdfa = f_gdfa.read().splitlines()

        for no, (line_pred, line_inter, line_gdfa) in enumerate(zip(lines_predicted, lines_intersection, lines_gdfa)):
            verse_id, aligns_predicted = line_pred.split('\t')
            _, aligns_intersection = line_inter.split('\t')
            _, aligns_gdfa = line_gdfa.split('\t')
            
            # Convert string alignments to set
            aligns_predicted =  set(aligns_predicted.split(' '))
            aligns_intersection = set(aligns_intersection.split(' '))
            aligns_gdfa = set(aligns_gdfa.split(' '))

            # combine base alignments with predictions
            aligns_NMF_plus_intersection = aligns_predicted.union(aligns_intersection)
            aligns_NMF_plus_gdfa = aligns_predicted.union(aligns_gdfa)

            # update results for all alignments
            calc_and_update_score(aligns_intersection, pros[verse_id], surs[verse_id], results_all["base_intersection"])
            calc_and_update_score(aligns_gdfa, pros[verse_id], surs[verse_id], results_all["base_gdfa"])
            calc_and_update_score(aligns_predicted, pros[verse_id], surs[verse_id], results_all["NMF"])
            calc_and_update_score(aligns_NMF_plus_intersection, pros[verse_id], surs[verse_id], results_all["NMF + intersection"])
            calc_and_update_score(aligns_NMF_plus_gdfa, pros[verse_id], surs[verse_id], results_all["NMF + gdfa"]) 

    with open(os.path.join(args.save_path, save_name), 'w') as f_out:
        for i in results_all:
            f_out.write(f'----{i}----\nPrecision: {results_all[i]["prec"]}\nRecall: {results_all[i]["rec"]}\nF1: {results_all[i]["f1"]}\nAER: {results_all[i]["aer"]}\nHits: {results_all[i]["total_hit_count"]}\n\n')
            print(f'----{i}----\nPrecision: {results_all[i]["prec"]}\nRecall: {results_all[i]["rec"]}\nF1: {results_all[i]["f1"]}\nAER: {results_all[i]["aer"]}\nHits: {results_all[i]["total_hit_count"]}\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/results/", type=str)
    parser.add_argument('--gold_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/pbc_utils/data/eng_fra_pbc/eng-fra.gold", type=str)    
    parser.add_argument('--predicted_alignments_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/predicted_alignments/predicted_alignments_from_eng-x-bible-mixed_to_fra-x-bible-louissegond_with_max_83_editions_for_250_verses_NMF.txt", type=str)    
    parser.add_argument('--intersection_alignments_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/predicted_alignments/intersection_alignments_from_eng-x-bible-mixed_to_fra-x-bible-louissegond_for_250_verses.txt", type=str)    
    parser.add_argument('--gdfa_alignments_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/predicted_alignments/gdfa_alignments_from_eng-x-bible-mixed_to_fra-x-bible-louissegond_for_250_verses.txt", type=str)    
    parser.add_argument('--source_edition', default="eng-x-bible-mixed", type=str) 
    parser.add_argument('--target_edition', default="fra-x-bible-louissegond", type=str) 

    args = parser.parse_args()
    main(args)