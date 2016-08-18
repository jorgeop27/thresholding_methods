#!/usr/bin/env python

import os
import argparse
import time
import re
import itertools
import numpy as np
import requests

organismsID = {'Escherichia_coli_ATCC_27325': '83333',
               'Mycobacterium_tuberculosis_ATCC_25618': '83332',
               'Bacillus_subtilis_168': '224308',
               'Ruegeria_pomeroyi': '246200'}

re_go = re.compile(r'GO:\d{7}')


def get_uniprot_data(species, gene):
    url_req = "http://www.uniprot.org/uniprot/?query=organism:%s+gene:%s&format=tab&columns=id,go" % (species, gene)
    # print 'URl Request: %s' % url_req
    try:
        r = requests.get(url_req)
    except requests.ConnectionError:
        return None, None
    if r.status_code != 200:
        print "Error!! UniProtDB returned a Error Code 200."
        print "URL Request: %s" % url_req
        return None, None
    else:
        if len(r.text) == 0:
            print "Error!! Empty response for Gene %s and Organism %s." % (gene, species)
            return None, None
        else:
            ln = r.text.split('\n')[1]
            values = ln.strip().split('\t')
            prot_id = values[0]
            if len(values) > 1:
                gos = re.findall(re_go, values[1])
            else:
                print "Protein %s with no GO Terms associated" % prot_id
                gos = []
            return prot_id, gos


def main(scores, labels, gotypes, fill_godata=True, restrict_labels=True, translate_file=None):
    # restrict_labels = True: Only use common GO terms between scores and labels
    # restrict_labels = False: Use all GO terms from the GO classification (Biological, Molecular or Cellular)
    # translate_file = None: Search for the gene name and GO terms online, in the UniProt catalogue
    # translate_file = filepath: Use the file as a translate b/w gene names and protein ID
    # fill_data = False: Do not complete the data of the GO terms in the labels
    # fill_data = True: Complete the data of the GO terms from the data obtained from translate_file.
    basedir, filename = os.path.split(scores)
    basename = os.path.splitext(filename)[0]
    
    bio_prots_file = os.path.join(basedir, '%s.biological_proteins' % basename)
    bio_got_file = os.path.join(basedir, '%s.biological_goterms' % basename)
    bio_scores_file = os.path.join(basedir, '%s.biological_scores' % basename)
    bio_labels_file = os.path.join(basedir, '%s.biological_labels' % basename)
    bio_raw_scores = os.path.join(basedir, '%s.biological_raw_scores' % basename)
    bio_raw_labels = os.path.join(basedir, '%s.biological_raw_labels' % basename)

    mol_prots_file = os.path.join(basedir, '%s.molecular_proteins' % basename)
    mol_got_file = os.path.join(basedir, '%s.molecular_goterms' % basename)
    mol_scores_file = os.path.join(basedir, '%s.molecular_scores' % basename)
    mol_labels_file = os.path.join(basedir, '%s.molecular_labels' % basename)
    mol_raw_scores = os.path.join(basedir, '%s.molecular_raw_scores' % basename)
    mol_raw_labels = os.path.join(basedir, '%s.molecular_raw_labels' % basename)
    
    cel_prots_file = os.path.join(basedir, '%s.cellular_proteins' % basename)
    cel_got_file = os.path.join(basedir, '%s.cellular_goterms' % basename)
    cel_scores_file = os.path.join(basedir, '%s.cellular_scores' % basename)
    cel_labels_file = os.path.join(basedir, '%s.cellular_labels' % basename)
    cel_raw_scores = os.path.join(basedir, '%s.cellular_raw_scores' % basename)
    cel_raw_labels = os.path.join(basedir, '%s.cellular_raw_labels' % basename)

    if translate_file is None:
        translate_file = os.path.join(basedir, 'geneID_proteinID_translate.%s' % basename)
        create_transfile = True
    else:
        create_transfile = False
    
    if os.path.exists(bio_prots_file) or os.path.exists(bio_got_file) or os.path.exists(bio_scores_file) or \
            os.path.exists(bio_labels_file) or os.path.exists(mol_prots_file) or os.path.exists(mol_got_file) or \
            os.path.exists(mol_scores_file) or os.path.exists(mol_labels_file) or os.path.exists(cel_prots_file) or \
            os.path.exists(cel_got_file) or os.path.exists(cel_scores_file) or os.path.exists(cel_labels_file):
        yn = raw_input("At least one output file already exist. Overwrite? y/n: ")
        if yn.upper() == 'N':
            return -1
    
    # Create GO Term types table:
    # Biological: 1
    # Molecular: 2
    # Cellular: 3
    print "GO Term typology..."
    t0 = time.time()
    goterm_type = {}
    with open(gotypes, 'r') as g:
        for ln in g:
            arr = ln.strip().split('\t')
            go = arr[0]
            gotype = arr[1]
            if gotype == 'biological_process':
                goterm_type[go] = 1
            elif gotype == 'molecular_function':
                goterm_type[go] = 2
            elif gotype == 'cellular_component':
                goterm_type[go] = 3
            else:
                print "GO Term %s Not Known..." % arr[1]
                continue
    t1 = time.time()
    print "Time: %.2f" % (t1 - t0)

    print "Initial pre-processing of labels..."
    print "Labels/GO Terms classification in biological/molecular/cellular processes"
    bio_protein_list_labels = {}    # List of proteins and their GO terms associated (biological processes)
    mol_protein_list_labels = {}
    cel_protein_list_labels = {}
    with open(labels, 'r') as main_labels:
        for ln in main_labels:
            values = ln.strip().split('\t')
            prot_id = values[0]
            goterm = values[1]
            gotype = goterm_type.get(goterm, None)
            if gotype is None:
                print "%s: Type not found" % goterm
            elif gotype == 1:
                if prot_id.upper() not in bio_protein_list_labels.keys():
                    bio_protein_list_labels[prot_id.upper()] = [goterm]
                else:
                    bio_protein_list_labels[prot_id.upper()].append(goterm)
            elif gotype == 2:
                if prot_id.upper() not in mol_protein_list_labels.keys():
                    mol_protein_list_labels[prot_id.upper()] = [goterm]
                else:
                    mol_protein_list_labels[prot_id.upper()].append(goterm)
            elif gotype == 3:
                if prot_id.upper() not in cel_protein_list_labels.keys():
                    cel_protein_list_labels[prot_id.upper()] = [goterm]
                else:
                    cel_protein_list_labels[prot_id.upper()].append(goterm)
            else:
                print "GO Term Type not recognized: %d" % gotype

    gene_protein_id = {}  # Map between gene names and protein IDs
    if create_transfile is True:
        translate_lines = []
    else:
        translate_lines = []
        with open(translate_file, 'r') as tf:
            for ln in tf:
                arr = ln.strip().split('\t')
                new_prot = arr[1]
                gene_protein_id[arr[0].upper()] = new_prot
                if fill_godata:
                    go_terms = arr[2:]
                    for new_go in go_terms:
                        gotype = goterm_type.get(new_go, None)
                        if gotype is None:
                            print "%s: Type not found" % new_go
                        elif gotype == 1:
                            if new_prot.upper() not in bio_protein_list_labels.keys():
                                bio_protein_list_labels[new_prot.upper()] = [new_go]
                            else:
                                bio_protein_list_labels[new_prot.upper()].append(new_go)
                        elif gotype == 2:
                            if new_prot.upper() not in mol_protein_list_labels.keys():
                                mol_protein_list_labels[new_prot.upper()] = [new_go]
                            else:
                                mol_protein_list_labels[new_prot.upper()].append(new_go)
                        elif gotype == 3:
                            if new_prot.upper() not in cel_protein_list_labels.keys():
                                cel_protein_list_labels[new_prot.upper()] = [new_go]
                            else:
                                cel_protein_list_labels[new_prot.upper()].append(new_go)
                        else:
                            print "GO Term Type not recognized: %d" % gotype

    if restrict_labels:
        biological_goterms = list(set(itertools.chain.from_iterable(bio_protein_list_labels.values())))
        molecular_goterms = list(set(itertools.chain.from_iterable(mol_protein_list_labels.values())))
        cellular_goterms = list(set(itertools.chain.from_iterable(cel_protein_list_labels.values())))
    else:
        biological_goterms = [go for go in goterm_type.keys() if goterm_type[go] == 1]
        molecular_goterms = [go for go in goterm_type.keys() if goterm_type[go] == 2]
        cellular_goterms = [go for go in goterm_type.keys() if goterm_type[go] == 3]

    print "Initial Labels/GO Terms classification done."
    t2 = time.time()
    print "Time: %.2f" % (t2 - t1)

    print "Initial pre-processing of scores..."

    bio_prots = []          # All the proteins with biological ontologies common to the labels and scores
    bio_prot_scores = []    # All the lines from the proteins above that belong to the biological ontology
    mol_prots = []
    mol_prot_scores = []
    cel_prots = []
    cel_prot_scores = []
    species = organismsID[basename]
    with open(scores, 'r') as main_scores:
        for ln in main_scores:      # ln is a line with the following format: Gene_Name\tGOTerm\tScore\n
            values = ln.strip().split('\t')
            genes_name = values[0].split('|')
            goterm = values[1]
            gotype = goterm_type.get(goterm, False)
            if gotype is None or gotype not in (1, 2, 3):
                print "%s type not found or not recognized" % goterm
                continue
            else:
                gene_name = genes_name[0]
                if gene_name.upper() not in gene_protein_id.keys():
                    prot_id, goterms = get_uniprot_data(species, gene_name)
                    if prot_id is not None:
                        gene_protein_id[gene_name.upper()] = prot_id.upper()
                        if create_transfile:
                            trline = [gene_name, prot_id]
                            if goterms is not None:
                                for goline in goterms:
                                    trline.append(goline)
                            translate_lines.append('\t'.join(trline) + '\n')
                        if gotype == 1:
                            if prot_id.upper() in bio_protein_list_labels.keys() and goterm in biological_goterms:
                                bio_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                bio_prot_scores.append(values)
                        elif gotype == 2:
                            if prot_id.upper() in mol_protein_list_labels.keys() and goterm in molecular_goterms:
                                mol_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                mol_prot_scores.append(values)
                        else:   # gotype == 3
                            if prot_id.upper() in cel_protein_list_labels.keys() and goterm in cellular_goterms:
                                cel_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                cel_prot_scores.append(values)
                    else:
                        gene_protein_id[gene_name.upper()] = None
                else:
                    prot_id = gene_protein_id[gene_name.upper()]
                    if prot_id is not None:
                        if gotype == 1:
                            if prot_id.upper() in bio_protein_list_labels.keys() and goterm in biological_goterms:
                                bio_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                bio_prot_scores.append(values)
                        elif gotype == 2:
                            if prot_id.upper() in mol_protein_list_labels.keys() and goterm in molecular_goterms:
                                mol_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                mol_prot_scores.append(values)
                        else:   # gotype == 3
                            if prot_id.upper() in cel_protein_list_labels.keys() and goterm in cellular_goterms:
                                cel_prots.append(prot_id.upper())
                                values[0] = prot_id.upper()
                                cel_prot_scores.append(values)
                    else:
                        continue
    if create_transfile:
        with open(translate_file, 'w') as trw:
            trw.writelines(translate_lines)
    print "Score's initial classification done..."
    t3 = time.time()
    print "Time: %.2f" % (t3 - t2)

    print "Biological processes:"
    print "Extracting proteins with biological GOs and possible GO terms (Labels)..."
    bio_prots = list(set(bio_prots))
    bio_prots_labels = bio_protein_list_labels.keys()
    for bprot in bio_prots_labels:
        if bprot not in bio_prots:
            print "Protein %s in label's list but not in the score's one (Biological)" % bprot
            del bio_protein_list_labels[bprot]
    bio_prots_labels = bio_protein_list_labels.keys()
    bio_prots.sort()
    bio_prots_labels.sort()
    assert bio_prots == bio_prots_labels

    biological_goterms.sort()

    # bio_prots are all the proteins that have a biological process (sorted)
    # biological_goterms are all the possible labels (GO terms) for biological processes (sorted)
    bio_prot_num = len(bio_prots)
    bio_labels_num = len(biological_goterms)
    print "Total number of proteins: %d\tTotal number of GO Terms: %d" % (bio_prot_num, bio_labels_num)

    bio_proteins = {pr: ind for ind, pr in enumerate(bio_prots)}
    bio_labs = {lbl: ind for ind, lbl in enumerate(biological_goterms)}

    with open(bio_prots_file, 'w') as bpf:
        for pr in bio_prots:
            bpf.write("%s\t%d\n" % (pr, bio_proteins[pr]))

    with open(bio_got_file, 'w') as bgf:
        for go in biological_goterms:
            bgf.write("%s\t%d\n" % (go, bio_labs[go]))

    t4 = time.time()
    print "Time: %.2f" % (t4 - t3)

    bio_scores = np.zeros((bio_prot_num, bio_labels_num), dtype='float64')
    bio_labels = np.zeros((bio_prot_num, bio_labels_num), dtype='uint8')

    print "Processing Biological Scores..."
    with open(bio_scores_file, 'w') as bioscr, open(bio_raw_scores, 'w') as biorws:
        for ln in bio_prot_scores:
            prot_index = bio_proteins.get(ln[0], False)
            got_index = bio_labs.get(ln[1], False)
            if prot_index is False or got_index is False:
                print "Error!! Protein %s or Go Term %s not found in the biological lists..." % (ln[0], ln[1])
            else:
                ln.extend([str(prot_index), str(got_index)])
                biorws.write('\t'.join(ln) + '\n')
                bio_scores[prot_index, got_index] = ln[2]
        np.savetxt(bioscr, bio_scores, delimiter=',')
    t5 = time.time()
    print "Done. Time: %.2f" % (t5 - t4)

    print "Processing Biological Labels..."
    with open(bio_labels_file, 'w') as biolbl, open(bio_raw_labels, 'w') as biorwl:
        for prot_id in bio_prots:
            bio_prot_goterms = bio_protein_list_labels.get(prot_id, [])
            if len(bio_prot_goterms) == 0:
                print "Error!! Protein %s not found in the biological lists..." % prot_id
            else:
                prot_index = bio_proteins.get(prot_id, False)
                for got in bio_prot_goterms:
                    got_index = bio_labs.get(got, False)
                    if prot_index is False or got_index is False:
                        print "Protein %s or Go term %s not in biological list..." % (prot_id, got)
                    else:
                        biorwl.write('\t'.join([prot_id, got, str(prot_index), str(got_index)]) + '\n')
                        bio_labels[prot_index, got_index] = 1
        np.savetxt(biolbl, bio_labels, fmt='%d', delimiter=',')
    t6 = time.time()
    print "Done. Time: %.2f" % (t6 - t5)

    print "Molecular processes:"
    print "Extracting proteins with molecular GOs and possible GO terms (Labels)..."
    mol_prots = list(set(mol_prots))
    mol_prots_labels = mol_protein_list_labels.keys()
    for mprot in mol_prots_labels:
        if mprot not in mol_prots:
            print "Protein %s in label's list but not in the score's one (Molecular)" % mprot
            del mol_protein_list_labels[mprot]
    mol_prots_labels = mol_protein_list_labels.keys()
    mol_prots.sort()
    mol_prots_labels.sort()
    assert mol_prots == mol_prots_labels

    molecular_goterms.sort()

    mol_prot_num = len(mol_prots)
    mol_labels_num = len(molecular_goterms)
    print "Total number of proteins: %d\tTotal number of GO Terms: %d" % (mol_prot_num, mol_labels_num)

    mol_proteins = {pr: ind for ind, pr in enumerate(mol_prots)}
    mol_labs = {lbl: ind for ind, lbl in enumerate(molecular_goterms)}

    with open(mol_prots_file, 'w') as mpf:
        for pr in mol_prots:
            mpf.write("%s\t%d\n" % (pr, mol_proteins[pr]))

    with open(mol_got_file, 'w') as mgf:
        for go in molecular_goterms:
            mgf.write("%s\t%d\n" % (go, mol_labs[go]))

    t7 = time.time()
    print "Time: %.2f" % (t7 - t6)

    mol_scores = np.zeros((mol_prot_num, mol_labels_num), dtype='float64')
    mol_labels = np.zeros((mol_prot_num, mol_labels_num), dtype='uint8')

    print "Processing Molecular Scores..."
    with open(mol_scores_file, 'w') as molscr, open(mol_raw_scores, 'w') as molrws:
        for ln in mol_prot_scores:
            prot_index = mol_proteins.get(ln[0], False)
            got_index = mol_labs.get(ln[1], False)
            if prot_index is False or got_index is False:
                print "Error!! Protein %s or Go Term %s not found in the molecular lists..." % (ln[0], ln[1])
            else:
                ln.extend([str(prot_index), str(got_index)])
                molrws.write('\t'.join(ln) + '\n')
                mol_scores[prot_index, got_index] = ln[2]
        np.savetxt(molscr, mol_scores, delimiter=',')
    t8 = time.time()
    print "Done. Time: %.2f" % (t8 - t7)

    print "Processing Molecular Labels..."
    with open(mol_labels_file, 'w') as mollbl, open(mol_raw_labels, 'w') as molrwl:
        for prot_id in mol_prots:
            mol_prot_goterms = mol_protein_list_labels.get(prot_id, [])
            if len(mol_prot_goterms) == 0:
                print "Error!! Protein %s not found in the molecular lists..." % prot_id
            else:
                prot_index = mol_proteins.get(prot_id, False)
                for got in mol_prot_goterms:
                    got_index = mol_labs.get(got, False)
                    if prot_index is False or got_index is False:
                        print "Protein %s or Go term %s not in molecular list..." % (prot_id, got)
                    else:
                        molrwl.write('\t'.join([prot_id, got, str(prot_index), str(got_index)]) + '\n')
                        mol_labels[prot_index, got_index] = 1
        np.savetxt(mollbl, mol_labels, fmt='%d', delimiter=',')
    t9 = time.time()
    print "Done. Time: %.2f" % (t9 - t8)

    print "Cellular processes:"
    print "Extracting proteins with cellular GOs and possible GO terms (Labels)..."
    cel_prots = list(set(cel_prots))
    cel_prots_labels = cel_protein_list_labels.keys()
    for cprot in cel_prots_labels:
        if cprot not in cel_prots:
            print "Protein %s in label's list but not in the score's one (Cellular)" % cprot
            del cel_protein_list_labels[cprot]
    cel_prots_labels = cel_protein_list_labels.keys()
    cel_prots.sort()
    cel_prots_labels.sort()
    assert cel_prots == cel_prots_labels

    cellular_goterms.sort()

    cel_prot_num = len(cel_prots)
    cel_labels_num = len(cellular_goterms)
    print "Total number of proteins: %d\tTotal number of GO Terms: %d" % (cel_prot_num, cel_labels_num)

    cel_proteins = {pr: ind for ind, pr in enumerate(cel_prots)}
    cel_labs = {lbl: ind for ind, lbl in enumerate(cellular_goterms)}

    with open(cel_prots_file, 'w') as cpf:
        for pr in cel_prots:
            cpf.write("%s\t%d\n" % (pr, cel_proteins[pr]))
    
    with open(cel_got_file, 'w') as cgf:
        for go in cellular_goterms:
            cgf.write("%s\t%d\n" % (go, cel_labs[go]))

    t10 = time.time()
    print "Time: %.2f" % (t10 - t9)

    cel_scores = np.zeros((cel_prot_num, cel_labels_num), dtype='float64')
    cel_labels = np.zeros((cel_prot_num, cel_labels_num), dtype='uint8')
    
    print "Processing Cellular Scores..."
    with open(cel_scores_file, 'w') as celscr, open(cel_raw_scores, 'w') as celrws:
        for ln in cel_prot_scores:
            prot_index = cel_proteins.get(ln[0], False)
            got_index = cel_labs.get(ln[1], False)
            if prot_index is False or got_index is False:
                print "Error!! Protein %s or Go Term %s not found in the cellular lists..." % (ln[0], ln[1])
            else:
                ln.extend([str(prot_index), str(got_index)])
                celrws.write('\t'.join(ln) + '\n')
                cel_scores[prot_index, got_index] = ln[2]
        np.savetxt(celscr, cel_scores, delimiter=',')
    t11 = time.time()
    print "Done. Time: %.2f" % (t11 - t10)

    print "Processing Cellular Labels..."
    with open(cel_labels_file, 'w') as cellbl, open(cel_raw_labels, 'w') as celrwl:
        for prot_id in cel_prots:
            cel_prot_goterms = cel_protein_list_labels.get(prot_id, [])
            if len(cel_prot_goterms) == 0:
                print "Error!! Protein %s not found in the cellular lists..." % prot_id
            else:
                prot_index = cel_proteins.get(prot_id, False)
                for got in cel_prot_goterms:
                    got_index = cel_labs.get(got, False)
                    if prot_index is False or got_index is False:
                        print "Protein %s or Go term %s not in cellular list..." % (prot_id, got)
                    else:
                        celrwl.write('\t'.join([prot_id, got, str(prot_index), str(got_index)]) + '\n')
                        cel_labels[prot_index, got_index] = 1
        np.savetxt(cellbl, cel_labels, fmt='%d', delimiter=',')
    t12 = time.time()
    print "Done. Time: %.2f" % (t12 - t11)
    print "Total process completed...."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data files got as output from the S2F Classifier.')
    parser.add_argument('scores_file', nargs=1, help='Filename of the data with the scores.')
    parser.add_argument('labels_file', nargs=1, help='Filename of the data with the labels.')
    parser.add_argument('--complete_go_data', action='store_true', dest='fill_godata', default=False,
                        help='If present, the script complete the labels data from the go data from the translate '
                             'file.')
    parser.add_argument('--restrict_labels', action='store_true', dest='restrict_labels', default=False,
                        help='If present, the script complete the labels data from the go data from the translate ')
    parser.add_argument('--translate_file', nargs='?', dest='translate_file', default=None,
                        help='If present, the script uses this file for translating gene names and protein IDs, and '
                             'also for completing GO data if --complete_go_data is also present.')
    args = parser.parse_args()
    scores_file = args.scores_file[0]
    labels_file = args.labels_file[0]
    fill_go_data = args.fill_godata
    restrict_lbls = args.restrict_labels
    trans_file = args.translate_file
    gotypes_file = '/Users/jorge/Documents/RHUL/Dissertation/Dataset/ontology_per_term'
    # print fill_go_data
    # print restrict_lbls
    # print trans_file
    init_time = time.time()
    main(scores_file, labels_file, gotypes_file, fill_go_data, restrict_lbls, trans_file)
    end_time = time.time()
    print "Total execution time (min): %.2f" % ((end_time - init_time) / 60)
