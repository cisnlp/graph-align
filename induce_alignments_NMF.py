from surprise import Dataset, Reader
import pandas as pd
from surprise import NMF
from utils import LOG, setup_dict_entry, load_editions, load_gold, get_verse_alignments
import os, argparse, random, math
from nltk.translate.gdfa import grow_diag_final_and
import numpy as np
from multiprocessing import Pool, Value


def get_row_col_editions(source_edition, target_edition, all_editions=None):
    row_editions = []
    col_editions = []
    for edition in all_editions:
        if edition != source_edition and edition != target_edition:
            row_editions.append(edition)
            col_editions.append(edition)

    row_editions.append(source_edition)
    col_editions.append(target_edition)

    return row_editions, col_editions

def get_aligns(rf, cf, alignments):
    raw_align = ''

    if rf in alignments and cf in alignments[rf]:
        raw_align = alignments[rf][cf]
        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[0]), int(x[1]) ) )
    elif cf in alignments and rf in alignments[cf]: # re: aak, ce: aai, 
        raw_align = alignments[cf][rf]
        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[1]), int(x[0]) ) )
    elif rf in alignments and rf == cf: # if source and target are the same
        keys = list(alignments[rf].keys())
        max_count = 0
        for key in keys:
            align = alignments[rf][key]
            for x in align.split():
                count = int(x.split('-')[0])
                if count > max_count:
                    max_count = count
        raw_align = "0-0"
        for i in range(1,max_count):
            raw_align += f" {i}-{i}"

        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[0]), int(x[1]) ) )
    else:
        return None
    
    return res

def add_aligns(aligns, aligns_dict, token_counts, re, ce, existing_items):
    for align in aligns:

        aligns_dict['userID'].append(re + str(align[0]))
        aligns_dict['itemID'].append(ce + str(align[1]))
        aligns_dict['rating'].append(3)

        if align[0] > token_counts[re]:
            token_counts[re] = align[0]
        if align[1] > token_counts[ce]:
            token_counts[ce] = align[1]
        
        existing_items[re][ce].append(f"{align[0]},{align[1]}")

def add_negative_samples(aligns_dict, existing_items, token_counts, verse_id):
    for re in existing_items:
        if token_counts[re] < 2:
            continue
        for ce in existing_items[re]:
            if token_counts[ce] < 2:
                continue
            for item in existing_items[re][ce]:
                i,j = tuple(item.split(","))
                i,j = (int(i), int(j))
                jp = random.randint(math.ceil(j+1), math.ceil(j+token_counts[ce] ))
                ip = random.randint(math.ceil(i+1), math.ceil(i+token_counts[re] ))

                jp %= (token_counts[ce] + 1)
                aligns_dict['userID'].append(re + str(i))
                aligns_dict['itemID'].append(ce + str(jp))
                aligns_dict['rating'].append(1)
                
                ip %= (token_counts[re] + 1) 
                aligns_dict['userID'].append(re + str(ip))
                aligns_dict['itemID'].append(ce + str(j))
                aligns_dict['rating'].append(1)

def get_alignments_df(row_editions, col_editions, verse_alignments,
         source_edition, target_edition, verse_id): #TODO can be improved a lot
    token_counts = {}
    existing_items = {}
    aligns_dict = {'itemID': [], 'userID': [], 'rating': []}
    for no, re in enumerate(row_editions):
        token_counts[re] = 0
        existing_items[re] = {}

        for ce in col_editions:

            setup_dict_entry(token_counts, ce, 0)
            existing_items[re][ce] = []
            aligns = get_aligns(re, ce, verse_alignments)
                
            if not aligns is None:
                add_aligns(aligns, aligns_dict, token_counts, re, ce, existing_items)
        
    add_negative_samples(aligns_dict, existing_items, token_counts, verse_id)

    return pd.DataFrame(aligns_dict), token_counts[source_edition], token_counts[target_edition]
    
def iter_max(sim_matrix: np.ndarray, max_count: int=2, alpha_ratio = 0.7) -> np.ndarray:
    m, n = sim_matrix.shape
    forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
    backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
    inter = forward * backward.transpose()

    if min(m, n) <= 2:
        return inter

    new_inter = np.zeros((m, n))
    count = 1
    while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            mask *= 0.0
            mask_zeros *= 0.0

        new_sim = sim_matrix * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
        new_inter = fwd * bac

        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1
    return inter
    
def get_itermax_predictions(raw_s_predictions, max_count=2, alpha_ratio=0.9):
    rows = len(raw_s_predictions)
    cols = len(raw_s_predictions[0])
    matrix = np.ndarray(shape=(rows, cols), dtype=float)

    for i in raw_s_predictions:
        for j, s in raw_s_predictions[i]:
            matrix[i,j] = s
    
    itermax_res = iter_max(matrix, max_count, alpha_ratio)
    res = []
    for i in range(rows):
        for j in range(cols):
            if itermax_res[i,j] != 0:
                res.append((i,j))
    
    return res

def predict_alignments(algo, source_edition, target_edition):
    raw_s_predictions = {}
    raw_t_predictions = {}

    for i in range(algo.s_tok_count + 1):
        for j in range(algo.t_tok_count + 1):
            pred = algo.predict(source_edition + str(i), target_edition + str(j))

            setup_dict_entry(raw_s_predictions, i, [])
            setup_dict_entry(raw_t_predictions, j, [])

            raw_s_predictions[i].append((j, pred.est))
            raw_t_predictions[j].append((i, pred.est))

    # get predicted alignments from argmax (max_count=1 means argmax)
    res = get_itermax_predictions(raw_s_predictions, max_count=1)

    return res

def train_model(df,  s_tok_count, t_tok_count, row_editions, col_editions):
    algo = NMF()
    reader = Reader(rating_scale=(1, 3))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    algo.s_tok_count = s_tok_count
    algo.t_tok_count = t_tok_count
    algo.row_editions = row_editions
    algo.col_editions = col_editions
    algo.df = df
    
    return algo

def get_induced_alignments(source_edition, target_edition, verse_alignments_path, verse_id, all_editions):

    verse_alignments = get_verse_alignments(verse_alignments_path, verse_id, editions=all_editions)
    
    # this is only for saving the gdfa alignments from source to target for the evauation
    verse_alignments_gdfa = get_verse_alignments(verse_alignments_path, verse_id, editions=[source_edition, target_edition], gdfa=True)

    ###  source -> row, target-> col###
    row_editions, col_editions = get_row_col_editions(source_edition, target_edition, all_editions)
    #itemid -> col, user -> row
    df, s_tok_count, t_tok_count = get_alignments_df(row_editions, col_editions, verse_alignments, source_edition, target_edition, verse_id)

    algo = train_model(df, s_tok_count, t_tok_count, row_editions, col_editions)
    
    predicted_alignments = predict_alignments(algo, source_edition, target_edition)
    base_inter_alignments = verse_alignments[source_edition][target_edition]
    base_gdfa_alignments = verse_alignments_gdfa[source_edition][target_edition]
    
    with cnt.get_lock():
        cnt.value += 1
        if cnt.value % 20 == 0:
            LOG.info(f"Done inferring alignments for {cnt.value} verses")

    return predicted_alignments, base_inter_alignments, base_gdfa_alignments,  len(algo.col_editions)+1


def init_globals(counter):
    global cnt
    cnt = counter

def main(args):
    random.seed(args.seed)

    pros, surs = load_gold(args.gold_file)
    all_verses =list(pros.keys())
    all_verses = all_verses

    # Get languages and editions
    editions, langs = load_editions(args.editions_file)
    all_editions = [editions[lang] for lang in langs]

    # print some info
    LOG.info(f"Inferring alignments from {args.source_edition} to {args.target_edition}")
    LOG.info(f"Number of verses whose alignments will be inferred: {len(all_verses)}")
    LOG.info(f"Number of editions to use for the graph algorithms: {len(all_editions)}")
    LOG.info(f"Number of cores to be used for processing: {args.core_count}")

    # Prepare arguments for parallel processing
    starmap_args = []
    for verse_id in all_verses:
        # aligns_predicted, used_edition_count = get_induced_alignments(args.source_edition, args.target_edition, args.verse_alignments, verse_id, all_editions)
        starmap_args.append((args.source_edition, args.target_edition, args.verse_alignments_path, verse_id, all_editions))

    # get predicted alignments using parallel processing
    cnt = Value('i', 0)
    with Pool(processes=args.core_count, initializer=init_globals, initargs=(cnt,)) as p:  
        all_alignments = p.starmap(get_induced_alignments, starmap_args)

    out_NMF_f_name = f"predicted_alignments_from_{args.source_edition}_to_{args.target_edition}_with_max_{len(all_editions)}_editions_for_{len(all_verses)}_verses_NMF.txt"
    out_NMF_file = open(os.path.join(args.save_path, out_NMF_f_name), 'w')
    out_inter_f_name = f"intersection_alignments_from_{args.source_edition}_to_{args.target_edition}_for_{len(all_verses)}_verses.txt"
    out_inter_file = open(os.path.join(args.save_path, out_inter_f_name), 'w')
    out_gdfa_f_name = f"gdfa_alignments_from_{args.source_edition}_to_{args.target_edition}_for_{len(all_verses)}_verses.txt"
    out_gdfa_file = open(os.path.join(args.save_path, out_gdfa_f_name), 'w')

    for id, verse_id in enumerate(all_verses):
        aligns_predicted, inter_aligns, gdfa_aligns, used_edition_count = all_alignments[id]

        # convert predicted alignments to string and write to a file
        aligns_predicted = ' '.join([f"{align[0]}-{align[1]}" for align in aligns_predicted])
        out_NMF_file.write(f"{verse_id}\t{aligns_predicted}\n")
        out_inter_file.write(f"{verse_id}\t{inter_aligns.strip()}\n")
        out_gdfa_file.write(f"{verse_id}\t{gdfa_aligns.strip()}\n")

    out_NMF_file.close()
    out_inter_file.close()
    out_gdfa_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/predicted_alignments", type=str)
    parser.add_argument('--gold_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/pbc_utils/data/eng_fra_pbc/eng-fra.gold", type=str)    
    parser.add_argument('--verse_alignments_path', default="/mounts/data/proj/ayyoob/align_induction/verse_alignments/", type=str)
    parser.add_argument('--source_edition', default="eng-x-bible-mixed", type=str) 
    parser.add_argument('--target_edition', default="fra-x-bible-louissegond", type=str) 
    parser.add_argument('--editions_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/pbc_utils/data/eng_fra_pbc/lang_list.txt", type=str)
    parser.add_argument('--core_count', default=80, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    main(args)
    
