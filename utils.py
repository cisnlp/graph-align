import logging
import os

def get_verse_alignments(verse_alignments_path, verse_id, editions=None, gdfa=False):

    f_path = verse_alignments_path + f"/{verse_id}"
    if gdfa:
        f_path += "_gdfa.txt"
    else:
        f_path += "_inter.txt"

    if not os.path.exists(f_path):
        LOG.info(f_path)
        LOG.info(f"=================================={verse_id} dos not exist==================================")
        return None

    res = {}
    with open(f_path, 'r') as f:
        for line in f:
            s_file, t_file, aligns = tuple(line.split('\t'))
            if (editions==None) or (s_file in editions and t_file in editions):
            	setup_dict_entry(res, s_file, {})
            	res[s_file][t_file] = aligns 
    return res

def calc_and_update_score(aligns, pros, surs, results):
    '''
    aligns: predicted alignments to be evaluated, can be:
        - string, eg "1-2 3-7 5-3"
        - set or list of strings, eg {'1-2', '3-7', '5-3'}
    pros: porbable alignments from gold alignments file (shown with 'p' instead of '-' in the gold file)
        - set of strings, eg {'1-2', '3-7', '5-3'}
    surs: sure alignments from the fold file 
        - set of strings, eg {'1-2', '3-7', '5-3'}
    results: a dictinary to be updated that contains the following keys:
        - p_hit_count
        - s_hit_count
        - total_hit_count
        - gold_s_hit_count
        - prec
        - rec
        - f1
        - aer
    '''

    if len(aligns) == 0: return None

    # match the type of 'aligns' to gold alignments, i.e. set of strings
    if type(aligns) == str:
        aligns = set(aligns.split(' '))

    # calculate # of hits
    p_hit = len(aligns & pros)
    s_hit = len(aligns & surs)
    total_hit = len(aligns)

    # Update hit counts
    results["p_hit_count"] += p_hit
    results["s_hit_count"] += s_hit
    results["total_hit_count"] += total_hit
    results["gold_s_hit_count"] += len(surs)

    # Update metrics
    results["prec"] = round(results["p_hit_count"] / max(results["total_hit_count"], 0.01), 3)
    results["rec"] = round(results["s_hit_count"] / results["gold_s_hit_count"], 3)
    results["f1"] = round(2. * results["prec"] * results["rec"] / max((results["prec"] + results["rec"]), 0.01), 3)
    results["aer"] = round(1 - (results["s_hit_count"] + results["p_hit_count"]) / (results["total_hit_count"] + results["gold_s_hit_count"]), 3)

def load_editions(editions_file):
    '''
    'editions_file' is a .txt file. Each line contains one language code (e.g. spa3)
    and name of the edition (e.g. spa-x-bible-newworld) separated by tab. 
    Language codes are unique for each edition so that different editions from one language
    can be used if desired.
    Other columns are discarded if there are any.
    '''

    editions = {}
    langs = []
    with open(editions_file) as f_lang_list:
        lines = f_lang_list.read().splitlines()
        for line in lines:
            comps = line.split('\t')
            editions[comps[0]] = comps[1]
            langs.append(comps[0])
    return editions, langs

def load_gold(g_path):
    '''
    loads gold alignments from the specified file as a dictionary.
    keys are the verse ids and values are alignments stored as sets of strings.
    '''

    gold_f = open(g_path, "r")
    pros = {}
    surs = {}

    for line in gold_f:
        line = line.strip().split("\t")
        line[1] = line[1].split()

        pros[line[0]] = set([x.replace("p", "-") for x in line[1]])
        surs[line[0]] = set([x for x in line[1] if "p" not in x])

    return pros, surs

def setup_dict_entry(_dict, entry, val):
	if entry not in _dict:
		_dict[entry] = val

def get_logger(name, filename, level=logging.DEBUG):
	logger = logging.getLogger(name)
	logger.setLevel(level)

	# fh = logging.FileHandler(filename)
	ch = logging.StreamHandler()

	# fh.setLevel(level)
	ch.setLevel(level)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# fh.setFormatter(formatter)
	ch.setFormatter(formatter)

	logger.addHandler(ch)
	# logger.addHandler(fh)

	return logger

LOG = get_logger("analytics", "logs/analytics.log")