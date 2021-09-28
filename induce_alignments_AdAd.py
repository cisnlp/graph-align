import os
import math
import codecs
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from utils import LOG, load_editions

class MyWG:
	def __init__(self, nodes=[]):
		self.nodes = nodes
		self.node_map = {x: i for i, x in enumerate(nodes)}
		self.es = defaultdict(dict)

	def add_nodes(self, nodes):
		for n in nodes:
			if n in self.node_map:
				print("Node already exist!", n)
		self.nodes.extend(nodes)
		self.node_map = {x: i for i, x in enumerate(nodes)}

	def add_edges(self, new_edges):
		for p in new_edges:
			if p[0] in self.nodes:
				self.es[self.node_map[p[0]]][self.node_map[p[1]]] = p[2]
			else:
				print("-------------\n\n no such nodes!!! \n\n\n", p)
				exit()

	def check_edges(self):
		base_nodes = [n for n in self.es]
		for n1 in base_nodes:
			for n2 in self.es[n1]:
				if n2 not in self.es or n1 not in self.es[n2]:
					self.es[n2][n1] = float(self.es[n1][n2])

	def calc_adar(self, edges, verbose=False):
		self.check_edges()

		scores = []
		for e in tqdm(edges, disable=not verbose):
			if e[0] in self.node_map:
				e = (self.node_map[e[0]], self.node_map[e[1]])

			score = 0.0
			for neib in set(self.es[e[0]]) & set(self.es[e[1]]):
				score += 1 / (math.log(len(self.es[neib])))
			scores.append(score)
		return scores

	def calc_wadar(self, edges, verbose=False):
		self.check_edges()

		scores = []
		for e in tqdm(edges, disable=not verbose):
			if e[0] in self.node_map:
				e = (self.node_map[e[0]], self.node_map[e[1]])

			score = 0.0
			for neib in set(self.es[e[0]]) & set(self.es[e[1]]):
				score += (self.es[e[0]][neib] + self.es[e[1]][neib]) / (math.log(1 + sum([self.es[neib][x] for x in self.es[neib]])))
			scores.append(score)
		return scores


def load_gold(gold_path="golds/eng-fra-new.gold"):
	golds = {}
	with open(gold_path, "r") as fi:
		for l in fi:
			l = l.split("\t")
			golds[l[0]] = list(set(l[1].split()))
	return golds

def load_texts_and_alignments(editions_file, lang_files_path, verse_alignments_path, aligner="inter", golds=None):
	# Get languages and editions
	editions, langs = load_editions(editions_file)
	all_editions = [editions[lang] for lang in langs]
	lang_pairs = [(l1, l2) for i, l1 in enumerate(langs) for j, l2 in enumerate(langs[i+1:])]
	lang_code_map = {editions[lang]: lang for lang in editions}

	texts = {}
	for langl in langs:
		verses = {}
		lang_path = lang_files_path + "/" + l[1] + ".txt"
		with codecs.open(lang_path, "r", "utf-8") as fi:
			for l in fi:
				if l[0] == "#": continue
				l = l.split("\t")
				if len(l) != 2 or len(l[1].strip()) == 0: continue
				if golds != None and l[0] not in golds: continue

				verses[l[0]] = [F"{langl}:{i}:{w}" for i, w in enumerate(l[1].split())]
		texts[langl] = verses

	if golds == None:
		golds = set(list(texts[langs[0]].keys()))

	init_aligns = defaultdict(dict)
	for verse in tqdm(golds):
		v_path = F"{verse_alignments_path}/{verse}_{aligner}.txt"
		if not os.path.exists(v_path):
			LOG.info(v_path)
			LOG.info(f"================================== dos not exist ==================================")
			return None
		with open(v_path, "r") as f_al:
			for vl in f_al:
				vl = vl.split("\t")
				if vl[0] not in all_editions or vl[1] not in all_editions:
					continue
				l1, l2 = lang_code_map[vl[0]], lang_code_map[vl[1]]
				is_reverse = False
				if (l1, l2) not in lang_pairs:
					l1, l2 = l2, l1
					is_reverse = True

				if not is_reverse:
					init_aligns[(l1, l2)][verse] = [[int(alp.split("-")[0]), int(alp.split("-")[1]), 1.0] for alp in vl[2].strip().split()]
				else:
					init_aligns[(l1, l2)][verse] = [[int(alp.split("-")[1]), int(alp.split("-")[0]), 1.0] for alp in vl[2].strip().split()]

	return langs, texts, lang_pairs, init_aligns

def get_alignment_matrix(sim_matrix):
	m, n = sim_matrix.shape
	forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
	backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
	return forward, backward.transpose()

def calc_adars(texts, lang_pairs, waligns, target_pair=("eng", "fra")):
	cur_e_scores = {}
	cur_e_wscores = {}
	new_e_scores = {}
	new_e_wscores = {}

	for verse in tqdm(texts[target_pair[0]]):
		all_nodes = []
		for lang in texts:
			if verse in texts[lang]:
				all_nodes.extend(texts[lang][verse])
		G = MyWG(all_nodes)

		missed_edges = []
		thresh_edges = []
		for lp in waligns:
			if lp not in lang_pairs: continue
			if verse not in waligns[lp] or len(waligns[lp][verse]) == 0: continue
			l1, l2 = lp[0], lp[1]

			try:
				edge_cover = set()
				for alp in waligns[lp][verse]:
					edge_cover.add((alp[0], alp[1]))
					G.add_edges([[texts[l1][verse][alp[0]], texts[l2][verse][alp[1]], alp[2]]])
			except:
				import pdb; pdb.set_trace()

			# calc score for missed_edges
			if lp == target_pair:
				for i, w1 in enumerate(texts[l1][verse]):
					for j, w2 in enumerate(texts[l2][verse]):
						if (i, j) not in edge_cover:
							missed_edges.append((w1, w2))
						else:
							thresh_edges.append((w1, w2))

		cur_e_scores[verse] = []
		cur_e_wscores[verse] = []
		new_e_scores[verse] = []
		new_e_wscores[verse] = []

		adar_scores = G.calc_adar(missed_edges)
		for sc, p in zip(adar_scores, missed_edges):
			if sc > 0:
				new_e_scores[verse].append([round(sc, 3), p])
		wadar_scores = G.calc_wadar(missed_edges)
		for sc, p in zip(wadar_scores, missed_edges):
			if sc > 0:
				new_e_wscores[verse].append([round(sc, 3), p])

		th_adar_scores = G.calc_adar(thresh_edges)
		for sc, p in zip(th_adar_scores, thresh_edges):
			if sc > 0:
				cur_e_scores[verse].append([round(sc, 3), p])

		th_wadar_scores = G.calc_wadar(thresh_edges)
		for sc, p in zip(th_wadar_scores, thresh_edges):
			if sc > 0:
				cur_e_wscores[verse].append([round(sc, 3), p])

		cur_e_scores[verse].sort(reverse=True)
		cur_e_wscores[verse].sort(reverse=True)
		new_e_scores[verse].sort(reverse=True)
		new_e_wscores[verse].sort(reverse=True)

	return new_e_scores, new_e_wscores, cur_e_scores, cur_e_wscores

def add_edges_to_align_argmax(texts, waligns, out_path="", target_pair=("eng", "fra"), cur_edges=None, new_edges=None):
	all_cnt = [0, 0]
	with open(out_path, "w") as f_new_al:
		for verse in texts[target_pair[0]]:
			if verse not in texts[target_pair[1]]: continue

			sim = np.zeros((len(texts[target_pair[0]][verse]), len(texts[target_pair[1]][verse])))

			final_aligns = set([F"{p[0]}-{p[1]}" for p in waligns[target_pair][verse]])
			all_cnt[0] += len(final_aligns)

			if len(new_edges[verse]) == 0:
				all_cnt[1] += len(final_aligns)
				final_aligns = " ".join(sorted(list(final_aligns)))
				f_new_al.write(F"{verse}\t{final_aligns}\n")
				continue

			lens = [len(target_pair[0]) + 1, len(target_pair[1]) + 1]
			for p in cur_edges[verse] + new_edges[verse]:
				e = (int(p[1][0][lens[0]: p[1][0][lens[0]:].find(":") + lens[0]]), int(p[1][1][lens[1]: p[1][1][lens[1]:].find(":") + lens[1]]))
				sim[e[0], e[1]] = p[0]

			fwd, rev = get_alignment_matrix(sim)
			sargmax = fwd * rev
			for i in range(sim.shape[0]):
				for j in range(sim.shape[1]):
					if sargmax[i, j] == 1:
						final_aligns.add(F"{i}-{j}")

			all_cnt[1] += len(final_aligns)
			final_aligns = " ".join(sorted(list(final_aligns)))
			f_new_al.write(F"{verse}\t{final_aligns}\n")

	return all_cnt

def main(args):
	target_pair = (args.source_lang, args.target_lang)
	if args.gold_file != "":
		pros, surs = load_gold(args.gold_file)
		all_verses = list(pros.keys())
	else:
		all_verses = None

	# Get languages and initial alignments
	langs, texts, lang_pairs, init_aligns = load_texts_and_alignments(args.editions_file, args.lang_files_path, args.verse_alignments_path, args.aligner, golds=all_verses)

	# print some info
	LOG.info(f"Inferring alignments from {args.source_lang} to {args.target_lang}")
	LOG.info(f"Number of verses whose alignments will be inferred: {len(all_verses)}")
	LOG.info(f"Number of editions to use for the graph algorithms: {len(langs)}")

	new_e_scores, new_e_wscores, cur_e_scores, cur_e_wscores = calc_adars(texts, lang_pairs, init_aligns, target_pair=target_pair)

	out_adad_file = os.path.join(args.save_path, F"{target_pair[0]}-{target_pair[1]}_adar.{args.aligner}")
	add_edges_to_align_argmax(texts, init_aligns, out_path=out_adad_file, target_pair=target_pair, cur_edges=cur_e_scores, new_edges=new_e_scores)

	out_wadad_file = os.path.join(args.save_path, F"{target_pair[0]}-{target_pair[1]}_wadar.{args.aligner}")
	add_edges_to_align_argmax(texts, init_aligns, out_path=out_wadad_file, target_pair=target_pair, cur_edges=cur_e_wscores, new_edges=new_e_wscores)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--save_path', default="/mounts/Users/cisintern/lksenel/Projects/pbc/graph-align/predicted_alignments", type=str)
	parser.add_argument('--gold_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/pbc_utils/data/eng_fra_pbc/eng-fra.gold", type=str)
	parser.add_argument('--verse_alignments_path', default="/mounts/data/proj/ayyoob/align_induction/verse_alignments/", type=str)
	parser.add_argument('--lang_files_path', default="/nfs/datc/pbc/", type=str)
	parser.add_argument('--source_lang', default="eng", type=str)
	parser.add_argument('--target_lang', default="fra", type=str)
	parser.add_argument('--editions_file', default="/mounts/Users/cisintern/lksenel/Projects/pbc/pbc_utils/data/eng_fra_pbc/lang_list.txt", type=str)
	parser.add_argument('--aligner', default="inter", type=str)

	args = parser.parse_args()
	main(args)
