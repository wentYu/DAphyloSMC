# DASMC - [module description]
# Copyright (C) 2026 Your Name / Your Institution
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
from multiprocessing import Pool

import numpy
from scipy.stats import gaussian_kde

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import random
from DASMC import p4
from DASMC.p4 import var,Mcmc,Chain
# import p4
# from p4.var import var
import scipy.stats
from scipy.special import logsumexp, gammaln
from ete3 import Tree, TreeStyle
import numpy as np

np.seterr(divide='ignore')
from scipy.linalg import expm
import pandas as pd
import dendropy
from dendropy.calculate import treecompare
import argparse
import re
import sumt
from joblib import load
import warnings

from DASMC.Tree2Vector import TreeToVector
from collections import OrderedDict

ATCG_dict = {'a': 0, 't': 1, 'c': 2, 'g': 3}

def calculate_Q_matrix(kappa, base_frequency):
    try:
        # under GTR, the kappa becomes the transform ratio list
        Q_origin = np.array([[-(
                    kappa[0] * base_frequency[1] + kappa[1] * base_frequency[2] + kappa[2] * base_frequency[3]),
                              kappa[0] * base_frequency[1], kappa[1] * base_frequency[2], kappa[2] * base_frequency[3]],
                             [kappa[0] * base_frequency[0], -(
                                         kappa[0] * base_frequency[0] + kappa[3] * base_frequency[2] + kappa[4] *
                                         base_frequency[3]), kappa[3] * base_frequency[2],
                              kappa[4] * base_frequency[3]],
                             [kappa[1] * base_frequency[0], kappa[3] * base_frequency[1], -(
                                         kappa[1] * base_frequency[0] + kappa[3] * base_frequency[1] + kappa[5] *
                                         base_frequency[3]), kappa[5] * base_frequency[3]],
                             [kappa[2] * base_frequency[0], kappa[4] * base_frequency[1], kappa[5] * base_frequency[2],
                              -(kappa[2] * base_frequency[0] + kappa[4] * base_frequency[1] + kappa[5] * base_frequency[
                                  2])]])
    except Exception as e:
        # under JC69 or K2P
        Q_origin = np.array([[-kappa - 2, 1, 1, kappa],
                             [1, -kappa - 2, kappa, 1],
                             [1, kappa, -kappa - 2, 1],
                             [kappa, 1, 1, -kappa - 2]])
    normalize = 0
    for i in range(len(Q_origin)):
        normalize -= base_frequency[i] * Q_origin[i][i]
    Q = Q_origin / normalize
    return Q

def calc_log_likelihood(node, length):
    # Calculate the log-likelihood of a tree with Felsenstein's dynamic programming using a recursive structure.
    if node.is_leaf():
        seq = node.sequence
        indices = np.array([ATCG_dict.get(ch, -1) for ch in seq], dtype=int)
        possibility = np.zeros((length, 4), dtype=float)
        valid = indices >= 0
        possibility[valid, indices[valid]] = 1.0
        missing = indices == -1
        possibility[missing] = 1.0
        return possibility, 0

    dist = 0
    possibility = np.ones((length, 4), dtype=float)
    for child in node.get_children():
        child_possibility, child_dist = calc_log_likelihood(child, length)
        dist += child_dist + child.dist
        possibility *= np.dot(child_possibility, np.exp(child.P).T)
    return possibility, dist

def compute_rcESS_log_domain(W, log_p, delta_phi):
    log_W = np.log(W)
    log_numerator_terms = log_W + delta_phi * log_p
    log_numerator_sum = logsumexp(log_numerator_terms)
    log_denominator_terms = log_W + 2 * delta_phi * log_p
    log_denominator_sum = logsumexp(log_denominator_terms)
    log_rcESS = 2 * log_numerator_sum - log_denominator_sum

    return np.exp(log_rcESS)

def compute_rcESS_log_domain_ref(W, log_f, log_rho, phi_r, phi, ref_bound):
    if phi == phi_r:
        return 1.0

    delta = phi - phi_r
    log_W = np.log(W)

    if phi <= ref_bound:
        log_factor = delta * (log_f - log_rho / ref_bound)
    elif phi_r >= ref_bound:
        log_factor = delta * log_f
    else:
        log_factor = delta * log_f + (phi_r / ref_bound - 1) * log_rho
    log_num = logsumexp(log_W + log_factor)
    log_den = logsumexp(log_W + 2 * log_factor)
    return np.exp(2 * log_num - log_den)


def get_leaf_set(node):
    return frozenset([leaf.taxon.label for leaf in node.leaf_iter()])

def find_threshold_vectorized(a, w, alpha):
    sorted_indices = np.argsort(a)[::-1]
    a_sorted = a[sorted_indices]
    w_sorted = w[sorted_indices]
    cumsum = np.cumsum(w_sorted)
    mask = cumsum >= alpha
    idx = np.where(mask)[0][0]
    smaller_mask = a_sorted < a_sorted[idx]
    if np.any(smaller_mask):
        return a_sorted[np.argmax(smaller_mask)]
    else:
        return a_sorted[idx] - 1

def dirichlet_proposal_compact(current, alpha=50.0):

    # Dirichlet proposal for evolutionary parameters

    curr = np.array(current)
    z = np.random.gamma(alpha * curr, 1.0)
    y = z / z.sum()

    shape_fwd = alpha * curr
    shape_rev = alpha * y

    new_log_q_rev_log_q_fwd = np.sum((shape_rev - 1) * np.log(current) - (shape_fwd - 1) * np.log(y)) + np.sum(
        gammaln(shape_fwd) - gammaln(shape_rev))

    return y.tolist(), new_log_q_rev_log_q_fwd

def read_nex_con_tre_file(dataset, mark, taxon_namespace, ts, d, gtr, primates_ref=False):
    taxa_labels = []
    if not primates_ref:
        if gtr:
            try:
                with open('./' + dataset + '/' + dataset + '_GTR.nex.con.tre', 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                with open('./' + dataset + '/' + dataset + '.nex.con.tre', 'r') as f:
                    lines = f.readlines()
        else:
            with open('./' + dataset + '/' + dataset + '.nex.con.tre', 'r') as f:
                lines = f.readlines()
    else:
        if gtr:
            try:
                with open('./primates/primates_ref_GTR.nex.con.tre', 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                with open('./primates/primates_ref.nex.con.tre', 'r') as f:
                    lines = f.readlines()
        else:
            with open('./primates/primates_ref.nex.con.tre', 'r') as f:
                lines = f.readlines()
    translate = {}
    i = 0
    for line in lines[-3 - len(d.taxNames):-4]:
        i += 1
        translate[str(i)] = line[4:-2].strip()
        taxa_labels.append(line[4:-2].strip())
    i += 1
    translate[str(i)] = lines[-4][4:-1].strip()
    taxa_labels.append(lines[-4][4:-1].strip())
    origin_mrbayes_newick = lines[-2][lines[-2].find('('):]

    dendro_mb = dendropy.Tree.get(
        data=origin_mrbayes_newick,
        schema="newick",
        preserve_underscores=True,
        taxon_namespace=dendropy.TaxonNamespace()
    )

    def convert_node(dendro_node, ete_parent=None):
        name = dendro_node.taxon.label if dendro_node.taxon else ""
        ete_node = Tree(name=name)

        prob_str = dendro_node.annotations.get_value('prob', None)
        if prob_str is not None:
            try:
                prob = float(prob_str)
                ete_node.add_feature("prob", prob)
                ete_node.support = prob
            except ValueError:
                pass

        if dendro_node.edge_length is not None:
            ete_node.dist = dendro_node.edge_length

        if ete_parent:
            ete_parent.add_child(ete_node)

        for child in dendro_node.child_nodes():
            convert_node(child, ete_node)
        return ete_node

    root = dendro_mb.seed_node
    ete_mb = convert_node(root)
    ete_mb.dist = 0

    for one in ete_mb.get_leaves():
        one.name = translate[one.name]
    try:
        ete_mb.sort_descendants()
    except Exception as e:
        e = 0

    ete_mb.render("./phylogenetic_tree_mb_" + dataset + "_" + mark + ".png", w=800, tree_style=ts)

    try:
        os.remove('./ete2dendropy_' + dataset + '_' + mark + '.newick')
    except Exception as e:
        e = 0
    ete_mb.write(outfile='ete2dendropy_' + dataset + '_' + mark + '.newick')

    mrbayes_dendro_consensus = dendropy.Tree.get_from_path('ete2dendropy_' + dataset + '_mark' + mark + '.newick',
                                                           'newick', taxon_namespace=taxon_namespace)
    mrbayes_ete_consensus = Tree('ete2dendropy_' + dataset + '_' + mark + '.newick')
    return mrbayes_ete_consensus, mrbayes_dendro_consensus


def save_posterior_para_distribution(log_likelihood_list, tree_length_list, base_frequency_list, kappa_list, dataset,
                                     mark, turn, gtr):
    param_names = [
        'log_likelihood',
        'tree_length',
        'pi(A)', 'pi(T)', 'pi(C)', 'pi(G)',
        'r(A<->T)', 'r(A<->C)', 'r(A<->G)', 'r(C<->T)', 'r(G<->T)', 'r(C<->G)'
    ]
    data = {}
    data['log_likelihood'] = np.array(log_likelihood_list)
    data['tree_length'] = np.array(tree_length_list)
    for i, name in enumerate(param_names[2:6]):

        data[name] = [base_frequency[i] for base_frequency in base_frequency_list]
    if gtr:
        for i, name in enumerate(param_names[6:]):

            data[name] = [kappa[i] for kappa in kappa_list]
    else:
        for i, name in enumerate(param_names[6:]):
            if i in [2, 3]:
                data[name] = [kappa for kappa in kappa_list]
            else:
                data[name] = [1 for kappa in kappa_list]

    samples_df = pd.DataFrame(data)
    samples_df = samples_df[param_names]
    samples_df.to_csv('./DASMC_output/' + dataset + "_" + mark + "_" + str(turn) + '_posterior_samples.csv', index=False)

    return

def propose(args):
    (rand, rand2, mcmc, kappa, base_frequency, chain, proposal_kappa_prob, proposal_pi_prob,
     gtr, prior_lambda, etbrPExt, eps, bps, iteration, k) = args

    feature = [0 for i in range(37)]
    log_proposal_ratio = None
    if rand > (proposal_kappa_prob + proposal_pi_prob):
        evolution = 0

        proposal = mcmc.props.chooseProposal(False)
        features = None
        new_kappa = kappa
        new_base_frequency = base_frequency
        if proposal.name == 'allBrLens':
            chain.proposeAllBrLens_Modified(proposal)
            if proposal_kappa_prob > 0:
                if not gtr:
                    feature = [0 for i in range(37)]
                else:
                    feature = [0 for i in range(45)]
            else:
                feature = [0 for i in range(35)]
        if proposal.name == 'NNI':
            proposal.brLenPriorType = 'exponential'
            proposal.brLenPriorLambda = prior_lambda
            proposal.etbrPExt = etbrPExt
            features = chain.proposeNNI_Blaise(proposal, extract_features=True)[1]

            if proposal_kappa_prob > 0:
                if not gtr:
                    features['Kappa_before'] = kappa
                    features['Kappa_after'] = kappa
                    feature = list(features.values())[:37]
                else:
                    for ti in range(5):
                        features['Kappa' + str(ti + 1) + '_before'] = kappa[ti]
                        features['Kappa' + str(ti + 1) + '_after'] = kappa[ti]
                    feature = list(features.values())[:45]
            else:
                feature = list(features.values())[:35]
        if proposal.name == 'eSPR':
            proposal.brLenPriorType = 'exponential'
            proposal.brLenPriorLambda = prior_lambda
            proposal.etbrPExt = etbrPExt
            features = chain.proposeESPR_Blaise_FromETBR(proposal, extract_features=True)[1]
            if proposal_kappa_prob > 0:
                if not gtr:
                    features['Kappa_before'] = kappa
                    features['Kappa_after'] = kappa
                    feature = list(features.values())[:37]
                else:
                    for ti in range(5):
                        features['Kappa' + str(ti + 1) + '_before'] = kappa[ti]
                        features['Kappa' + str(ti + 1) + '_after'] = kappa[ti]
                    feature = list(features.values())[:45]
            else:
                feature = list(features.values())[:35]
    else:
        if rand2 < (proposal_kappa_prob / (proposal_kappa_prob + proposal_pi_prob)):
            evolution = 1
            proposal = 'Kappa'
            new_base_frequency = base_frequency
            if not gtr:
                rand = scipy.stats.norm.rvs(loc=0, scale=0.2, size=1)[0]
                new_kappa = kappa * np.exp(rand)
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    new_kappa, log_proposal_ratio = dirichlet_proposal_compact(kappa, alpha=eps)

                    if w:
                        warn = False
                        for warning in w:
                            if warning.category == RuntimeWarning:
                                warn = True
                        # if warn:
                        #     print('RuntimeWarning:')
                        #     print(f'iteration={iteration},k={k}')
                        #     print(
                        #         f'old kappa={kappa}, new kappa={new_kappa}, log proposal ratio={log_proposal_ratio}')
        else:
            proposal = 'Pi'
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                new_kappa = kappa
                new_base_frequency, log_proposal_ratio = dirichlet_proposal_compact(base_frequency, alpha=bps)
            evolution = new_base_frequency
        pTree = chain.propTree
        for n in pTree.iterNodesNoRoot():
            n.br.oldLen = n.br.len
            n.br.oldNode = n
        y0 = None
        while not y0:
            nNum = random.choice(pTree.preOrder)
            if nNum != var.NO_ORDER and nNum != pTree.root.nodeNum:
                y0 = pTree.node(nNum)
        x0 = y0.parent
        myRan = random.random()
        if myRan >= 0.5:
            x0 = y0
            y0 = x0.parent
        features = {}
        features['feature_mode'] = 1
        features['Total_brLens'] = pTree.getLenMirror(c=0)
        features['Max_brLen'] = pTree.getMaxLen()
        features['Var_brLen'] = pTree.getVarLen()
        try:
            features['Prune_brLen'] = y0.br.len
        except Exception as e:
            features['Prune_brLen'] = 0
        features['Regraft_brLen'] = 0
        features['Prune_regraft_brLen_ratio'] = -1

        features['Prune2Regraft_leafNum'] = 0
        features['Prune2Regraft_total_brLens'] = 0
        features['Prune2Regraft_MaxbrLen'] = 0
        features['Prune2Regraft_VarbrLen'] = 0

        features['Prune2Regraft_nodeNum'] = 0
        features['Prune2Regraft_brLen'] = 0
        features['Prune2Regraft_Max_brLen'] = 0
        features['Prune2Regraft_Var_brLen'] = 0

        features['Subtree1_leafNum'] = len(list(y0.iterLeaves()))
        features['Subtree2_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))
        features['Subtree3_leafNum'] = 0
        features['Subtree4_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))

        features['Subtree1_total_brLens'], features['Subtree2_total_brLens'] = pTree.getPartLen(y0)
        features['Subtree3_total_brLens'] = 0
        features['Subtree4_total_brLens'] = features['Subtree2_total_brLens']
        features['Subtree1/3_brLens_ratio'] = -1

        features['Subtree1_max_brLen'], features['Subtree2_max_brLen'] = pTree.getPartMaxLen(y0)
        features['Subtree3_max_brLen'], features['Subtree4_max_brLen'] = 0, features['Subtree2_max_brLen']

        features['Subtree1_var_brLens'], features['Subtree2_var_brLens'], _, br_list2 = pTree.getPartVarLen(
            y0)
        features['Subtree3_var_brLens'], br_list1 = 0, []
        unique_vals, counts = np.unique(br_list1, return_counts=True)
        freq_dict = dict(zip(unique_vals, counts))
        mask = numpy.ones(len(br_list2), dtype=bool)
        for i, val in enumerate(br_list2):
            if val in freq_dict and freq_dict[val] > 0:
                mask[i] = False
                freq_dict[val] -= 1
        if len(mask) == 0 or mask.max() == False:
            res4 = 0
        else:
            res4 = numpy.var(br_list2[mask])
        features['Subtree4_var_brLens'] = res4
        features['Large2LongRisk'] = features['Subtree1_leafNum'] * features['Regraft_brLen']
        features['Reshape_brLens'] = 0
        features['Reshape_mul_brLens'] = 0
        if proposal_kappa_prob > 0:
            if not gtr:
                features['Kappa_before'] = kappa
                features['Kappa_after'] = new_kappa
                feature = list(features.values())[:37]
            else:
                for ti in range(5):
                    features['Kappa' + str(ti + 1) + '_before'] = kappa[ti]
                    features['Kappa' + str(ti + 1) + '_after'] = new_kappa[ti]
                feature = list(features.values())[:45]
        else:
            feature = list(features.values())[:35]

    return (
    evolution, proposal, log_proposal_ratio, feature, chain.logPriorRatio, chain.logProposalRatio, chain.propTree)


def calculate(args):
    (rand_acceptance, evolution, chain, kappa, base_frequency,
     prior_lambda, features, Q, name_seq_dict, newick_tree, gtr, gamma, check, length, log_proposal_ratio_list,
     reference, rho_x, RF, reference_ete_consensus, reference_bound, new_Phi, iteration, k,
     this_sur_log_likelihood_ratio, log_likelihood, ML_log_likelihood, proposal, K, log_prior, nFP) = args
    # due to the parallelization, the idea of a global cache is hard to realize now. But fortunately parallelization
    # seems to reduce more time than cache when using 3 or more cores.
    use_P_Q_dict = False
    P_Q_dict = None
    kappa_tuple = None
    new_kappa_tuple = None
    base_frequency_tuple = None
    new_base_frequency_tuple = None
    if check:
        if not evolution:
            new_newick_tree = Tree(chain.propTree.writeNewick(toString=True, spaceAfterComma=False))
            for node in list(new_newick_tree.traverse())[1:]:
                if use_P_Q_dict:
                    if gtr:

                        kappa_item = kappa_tuple
                    else:
                        kappa_item = kappa

                    base_frequency_item = base_frequency_tuple
                    P = P_Q_dict.get((node.dist, kappa_item, base_frequency_item), [None])

                    if P[0] is None:
                        P = np.log(expm(node.dist * Q))

                        P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = P
                        if len(P_Q_dict) > 300:
                            P_Q_dict.popitem(last=False)
                else:
                    P = np.log(expm(node.dist * Q))

                node.add_feature('P', P)
                if node.is_leaf():
                    node.add_feature('sequence', name_seq_dict.get(node.name))
            log_prior_ratio = chain.logPriorRatio
            log_proposal_ratio = chain.logProposalRatio
            new_base_frequency = base_frequency
        else:
            new_newick_tree = newick_tree
            if evolution == 1:
                if not gtr:
                    new_kappa = features[-1]
                    new_log_kappa_prior = np.log(gamma.pdf(new_kappa))
                    old_log_kappa_prior = np.log(gamma.pdf(kappa))
                    log_prior_ratio = new_log_kappa_prior - old_log_kappa_prior
                    log_proposal_ratio = np.log(new_kappa) - np.log(kappa)
                else:
                    new_kappa = features[-9::2]
                    new_kappa.append(1 - np.sum(new_kappa))
                    log_prior_ratio = 0
                    log_proposal_ratio = log_proposal_ratio_list[k]
                new_base_frequency = base_frequency
            else:
                log_prior_ratio = 0
                log_proposal_ratio = log_proposal_ratio_list[k]
                new_kappa = kappa
                new_base_frequency = evolution

            new_Q = calculate_Q_matrix(new_kappa, new_base_frequency)
            for node in list(new_newick_tree.traverse())[1:]:

                node.P = np.log(expm(node.dist * new_Q))
                if use_P_Q_dict:
                    if gtr:

                        kappa_item = new_kappa_tuple
                    else:
                        kappa_item = new_kappa

                    base_frequency_item = new_base_frequency_tuple
                    P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = node.P.copy()

        result = calc_log_likelihood(new_newick_tree, length)
        probability_among_sites = np.dot(result[0], np.array(new_base_frequency).T)
        new_log_likelihood = np.sum(np.log(probability_among_sites))

        if reference:
            log_reference_ratio = (np.log(rho_x(RF))
                                   - np.log(
                        rho_x(reference_ete_consensus.robinson_foulds(new_newick_tree, unrooted_trees=True)[0])))
            log_ref_ratio_with_exp = min(0, new_Phi / reference_bound - 1) * log_reference_ratio
        else:
            log_ref_ratio_with_exp = 0

    flag = False
    try:
        proposal_name = proposal.name
    except Exception as e:
        if evolution == 1:
            proposal_name = 'kappa'
        else:
            proposal_name = 'pi'
    should_acc = -1
    sur_log_likelihood_ratio = np.inf
    if check:
        should_acc = 0
        real_ratio = new_Phi * (
                    new_log_likelihood - log_likelihood) + log_prior_ratio + log_proposal_ratio + log_ref_ratio_with_exp
        if np.exp(real_ratio) >= rand_acceptance:
            should_acc = 1


    if proposal_name == 'allBrLens' or proposal_name == 'pi':
        this_sur_log_likelihood_ratio = np.inf
    if this_sur_log_likelihood_ratio >= K:

        if not evolution:
            new_newick_tree = Tree(chain.propTree.writeNewick(toString=True, spaceAfterComma=False))

            log_prior_ratio = chain.logPriorRatio

            log_proposal_ratio = chain.logProposalRatio
            new_base_frequency = base_frequency

        else:
            new_newick_tree = newick_tree

            if evolution == 1:
                new_base_frequency = base_frequency
                if not gtr:
                    new_kappa = features[-1]
                    new_log_kappa_prior = np.log(gamma.pdf(new_kappa))
                    old_log_kappa_prior = np.log(gamma.pdf(kappa))
                    log_prior_ratio = new_log_kappa_prior - old_log_kappa_prior
                    log_proposal_ratio = np.log(new_kappa) - np.log(kappa)
                else:
                    new_kappa = features[-9::2]
                    new_kappa.append(1 - np.sum(new_kappa))
                    new_log_kappa_prior = 0
                    old_log_kappa_prior = 0
                    log_prior_ratio = 0

                    log_proposal_ratio = log_proposal_ratio_list[k]

            else:
                new_kappa = kappa
                log_prior_ratio = 0

                log_proposal_ratio = log_proposal_ratio_list[k]

                new_base_frequency = evolution

            new_Q = calculate_Q_matrix(new_kappa, new_base_frequency)

        our_log_likelihood_ratio_1 = (this_sur_log_likelihood_ratio + log_likelihood) - min(
            log_likelihood, ML_log_likelihood)

        if reference:
            log_reference_ratio = (np.log(rho_x(RF))
                                   - np.log(
                        rho_x(reference_ete_consensus.robinson_foulds(new_newick_tree, unrooted_trees=True)[0])))
            log_ref_ratio_with_exp = min(0, new_Phi / reference_bound - 1) * log_reference_ratio

        else:
            log_reference_ratio = 0
            log_ref_ratio_with_exp = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if this_sur_log_likelihood_ratio == np.inf:
                our_ratio_1 = np.inf
            else:
                our_ratio_1 = log_prior_ratio + log_proposal_ratio + our_log_likelihood_ratio_1 * new_Phi + log_ref_ratio_with_exp

            if w:
                warn = False
                for warning in w:
                    if warning.category == RuntimeWarning:
                        warn = True
                # if warn:
                #     print('RuntimeWarning:')
                #     print(f'iteration={iteration},k={k}')
                #     print(
                #         f'log_prior_ratio={log_prior_ratio}, log_proposal_ratio={log_proposal_ratio}, proposal_name={proposal}')

        if our_ratio_1 > 0:
            our_ratio_1 = 0

        if (rand_acceptance <= np.exp(our_ratio_1)
                or proposal_name == 'allBrLens' or proposal_name == 'pi'):

            if not evolution:
                for node in list(new_newick_tree.traverse())[1:]:

                    if use_P_Q_dict:
                        if gtr:

                            kappa_item = kappa_tuple
                        else:
                            kappa_item = kappa

                        base_frequency_item = base_frequency_tuple
                        P = P_Q_dict.get((node.dist, kappa_item, base_frequency_item), [None])

                        if P[0] is None:
                            P = np.log(expm(node.dist * Q))

                            P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = P

                            if len(P_Q_dict) > 300:
                                P_Q_dict.popitem(last=False)
                    else:
                        P = np.log(expm(node.dist * Q))

                    node.add_feature('P', P)
                    if node.is_leaf():
                        node.add_feature('sequence', name_seq_dict.get(node.name))
            else:
                for node in list(new_newick_tree.traverse())[1:]:

                    node.P = np.log(expm(node.dist * new_Q))
                    if use_P_Q_dict:
                        if gtr:
                            kappa_item = new_kappa_tuple
                        else:
                            kappa_item = new_kappa

                        base_frequency_item = new_base_frequency_tuple
                        P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = node.P.copy()

            result = calc_log_likelihood(new_newick_tree, length)
            probability_among_sites = np.dot(result[0], np.array(new_base_frequency).T)
            new_log_likelihood = np.sum(np.log(probability_among_sites))

            our_calc_log_prior = -prior_lambda * result[1]

            our_log_prior_ratio = our_calc_log_prior - log_prior

            # check for prior
            # if abs(our_log_prior_ratio - log_prior_ratio) >= 0.1 and not evolution:
            #     print(proposal_name, k)
            #     print("ERROR: Different prior:", our_log_prior_ratio, log_prior_ratio)
            #     print(
            #         f'Our:{our_calc_log_prior},{log_prior}, Their:{chain.propTree.getLenMirror(c=0)},{chain.curTree.getLenMirror(c=0)}')

            if not nFP or new_log_likelihood / log_likelihood >= min(new_log_likelihood,
                                                                     log_likelihood +
                                                                     this_sur_log_likelihood_ratio) / min(
                log_likelihood,
                ML_log_likelihood):

                our_log_likelihood_ratio_2 = min(this_sur_log_likelihood_ratio + log_likelihood,
                                                 new_log_likelihood) - min(
                    log_likelihood, ML_log_likelihood)

                our_ratio_2 = log_prior_ratio + log_proposal_ratio + our_log_likelihood_ratio_2 * new_Phi + log_ref_ratio_with_exp

                if our_ratio_2 > 0:
                    our_ratio_2 = 0
                if rand_acceptance <= np.exp(our_ratio_2):

                    sur_log_likelihood_ratio = this_sur_log_likelihood_ratio
                    ML_log_likelihood = log_likelihood + this_sur_log_likelihood_ratio

                    if not evolution:
                        log_prior += log_prior_ratio
                    else:
                        if evolution == 1:
                            if not gtr:
                                kappa = new_kappa
                            else:
                                kappa = new_kappa.copy()
                        else:
                            base_frequency = new_base_frequency.copy()
                        Q = new_Q
                    flag = True
                    dict_no = -1
                else:
                    dict_no = 1
            else:
                dict_no = 0
        else:
            dict_no = 2
    else:
        dict_no = 3

    return (
    flag, log_likelihood, proposal_name, newick_tree, should_acc, dict_no, sur_log_likelihood_ratio, ML_log_likelihood,
    log_prior, kappa, base_frequency, Q)


def propose_parallel(pool, data):
    return pool.map(propose, data, chunksize=10)


def calculate_parallel(pool, data):
    return pool.map(calculate, data, chunksize=10)


def SMC(kappa, alpha, base_frequency, prior_lambda=10, etbrPExt=0.6, proposal_kappa_prob=0.02, proposal_pi_prob=0.02,
        brLen_prob=0.5,
        nParticles=50, iterations=20000, dataset='primates', mark='0', feature=0, output_feature=0, output_vector=0,
        self_adaptive=1,
        K=-50, delta=5, check=0, reference=0, pilotRF=0.0, turn=0, gtr=0, ga=0, eps=50.0, bps=50.0, use_P_Q_dict=1,
        reference_bound=1.0,
        nFP=0, Final_Resample=0, random_seed=0, duplicate_iterations=1, pool=None,pc=1):
    print('\nsettings:')

    print('\n--global settings--')
    print('a pilot run or a formal DA-SMC run (1 for pilot run)? ', feature)
    if feature:
        print('need to output the features? ', bool(output_feature))
    print('run on which dataset (your input file should be exactly "dataset.nex"): ', dataset)
    print('identifier for this set of works: ', mark)
    print('which turn is it in the repeated run (Optional, usually used in formal DA-SMC, not pilot run)? ', turn)
    print('random seed (0 means using default seeds): ', random_seed)
    print('open the Developer check mode? ', check)

    print('\n--proposal settings--')
    print('use model SYM or GTR: ', bool(gtr))
    print('probability of evolution rate parameters proposal (set to 0 if use model JC69): ', proposal_kappa_prob)
    print('probability of base frequency parameters proposal: ', proposal_pi_prob)
    if gtr == 0:
        print('initial kappa (set to 1 if use model JC69): ', kappa)
        if proposal_kappa_prob > 0:
            print('use informative prior: ', bool(ga))
    else:
        print('evolution rate parameters Direchlet proposal sharpness: ', eps)
    print('base frequency parameters Direchlet proposal sharpness: ', bps)
    print('ratio of branch length proposal and any topology proposal: ', brLen_prob)

    print('\n--SMC settings--')
    print('nucleobase frequency (A,T,C,G): ', base_frequency)
    print('branch length exponential prior lambda: ', prior_lambda)
    print('eSPR extend probability: ', etbrPExt)
    if feature:
        self_adaptive=0
        print('We do not recommend a self-adaptive pilot run due to its uncertainty in length.')
    print('use self-adaptive SMC: ', bool(self_adaptive))
    if self_adaptive:
        print('self-adaptive SMC temperature controller alpha: ', alpha)
    print('numbers of particles in one iteration: ', nParticles)
    print('total iterations (if not self-adaptive): ', iterations)

    print('\n--Delay Acceptance SMC settings--')
    print('first rejection threshold K: ', K)
    K += delta
    print('likelihood tuning bias δ: ', delta)
    print('no False Positive for DASMC kernel (Warning: activate it only with a large δ and low K): ', bool(nFP))

    print('\n--Advanced settings--')
    print('use a reference distribution: ', bool(reference))
    if not reference:
        reference_bound = 0
    else:
        print('reference distribution flatness: ', pilotRF)
        print('the scope of impact of reference (from 0 to 1): ', reference_bound)
    print('use a final correction for DASMC: ', bool(Final_Resample))
    print('cores used for parallelization: ', pc)
    print('output all trees to vectors (developing option, do not use now...): ', bool(output_vector))
    print('Waste-free SMC having duplicate MCMC moves sharing one SMC temperature (developing option, do not use now...): ',
          duplicate_iterations > 1)

    print('\nsettings ends...')


    if not feature:
        try:
            all_model = load('./RF_Output/RF_model_' + dataset +'_'+mark+ '.joblib')
        except Exception as e:
            raise RuntimeError('Have no model now, Provide one or run the pilot run at first.')
    sur_log_likelihood_ratio_list = np.array([np.inf for i in range(nParticles)])
    if not gtr:
        if ga:
            gamma = scipy.stats.gamma(a=2 * kappa, scale=0.5)
        else:
            gamma = scipy.stats.lomax(c=1, scale=1)
    else:
        gamma = None
    p4.var.doCheckForAllGapColumns = False
    p4.read('./data/' + dataset + '.nex')

    d = p4.Data()
    a = d.alignments[0]
    name_seq_dict = {}
    for sequence in a.sequences:
        name_seq_dict[sequence.name] = sequence.sequence
    length = len(sequence.sequence)

    if feature:
        reference = 0
    taxon_namespace = None
    reference_ete_consensus = None
    if not reference:
        rho_x = 1
    else:
        rho_x = lambda x: 1 / ((((x + 1) ** 2) - 1) * int(x > pilotRF) + 1)
        try:
            all_taxa = sorted([one.replace('_', ' ') for one in list(name_seq_dict.keys())])
            taxon_namespace = dendropy.TaxonNamespace(all_taxa)
            reference_dendro_consensus = dendropy.Tree.get_from_path(
                "./pilot_output/pilot_mcmc_consensus_" + dataset + "_" + mark + "_" + str(1) + ".con",
                "nexus", taxon_namespace=taxon_namespace)
            reference_ete_consensus = Tree(reference_dendro_consensus.__str__())
        except Exception as e:
            print("Haven't get reference yet! Inactivate reference distribution.")
            rho_x = lambda x: 1
            reference = 0

    log_weight_list = np.array([0.0 for k in range(nParticles)])
    weight_sum = 0
    newick_tree_list = [-1 for k in range(nParticles)]
    log_likelihood_list = np.array([0.0 for k in range(nParticles)])
    ML_log_likelihood_list = np.array([0.0 for k in range(nParticles)])
    log_prior_list = np.array([0.0 for k in range(nParticles)])
    if not gtr:
        kappa_list = np.array([float(kappa) for k in range(nParticles)])
    else:
        kappa_list = np.array([kappa for k in range(nParticles)])
    base_frequency_list = np.array([base_frequency for k in range(nParticles)])
    Q_list = [0.0 for k in range(nParticles)]
    for k in range(nParticles):
        Q_list[k] = calculate_Q_matrix(kappa_list[k], base_frequency_list[k])
    Phi_list = [0]
    weight_sum_list = []
    mcmc_list = []
    chain_list = []
    vector_list = []
    P_Q_dict = OrderedDict()
    print('start particles initialization...')
    for k in range(nParticles):
        weight_sum += np.exp(log_weight_list[k])
        t = p4.func.randomTree(taxNames=d.taxNames)
        t.data = d
        A = t.newComp(partNum=0, free=0, spec='equal')
        t.newRMatrix(partNum=0, free=0, spec='2p', val=kappa)
        t.setNGammaCat(partNum=0, nGammaCat=1)
        t.setPInvar(partNum=0, free=0, val=0.0)
        t.setModelComponentOnNode(A, node=t.root, clade=1)

        newick_tree_list[k] = Tree(t.writeNewick(toString=True, spaceAfterComma=False))
        if output_vector:
            vector_list.append(TreeToVector(newick_tree_list[k]))
        mcmc_list.append(Mcmc(t, nChains=1, runNum=0, sampleInterval=500, checkPointInterval=250000))
        mcmc_list[k].prob.brLen = 0
        mcmc_list[k].prob.allBrLens = brLen_prob
        mcmc_list[k].prob.local = 0
        mcmc_list[k].prob.eTBR = 0
        mcmc_list[k].prob.NNI = 1
        mcmc_list[k].prob.eSPR = 1
        mcmc_list[k]._makeProposals()
        chain_list.append(Chain(mcmc_list[k]))
        chain_list[k].curTree = t.dupe()
        chain_list[k].curTree.data = d
        chain_list[k].propTree = chain_list[k].curTree.dupe()
        chain_list[k].propTree.data = d

        for node in list(newick_tree_list[k].traverse())[1:]:
            if use_P_Q_dict:
                if gtr:
                    kappa_item = tuple(kappa_list[k])
                else:
                    kappa_item = kappa_list[k]
                base_frequency_item = tuple(base_frequency_list[k])
                P = P_Q_dict.get((node.dist, kappa_item, base_frequency_item), [None])
                if P[0] is None:
                    P = np.log(expm(node.dist * Q_list[k]))
                    P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = P
                    if len(P_Q_dict) > 300:
                        P_Q_dict.popitem(last=False)
            else:
                P = np.log(expm(node.dist * Q_list[k]))
            node.add_feature('P', P)
            if node.is_leaf():
                node.add_feature('sequence', name_seq_dict.get(node.name))

        result = calc_log_likelihood(newick_tree_list[k], length)
        probability_among_sites = np.dot(result[0], base_frequency_list[k].T)
        log_likelihood_list[k] = np.sum(np.log(probability_among_sites))
        ML_log_likelihood_list[k] = log_likelihood_list[k]
        log_prior_list[k] = -prior_lambda * result[1]

    normalized_weight_list = np.divide(np.exp(log_weight_list), weight_sum)

    all_vector_list_X = []
    all_vector_list_Y = []
    feature_output_list = []
    if feature:
        proposal_dict = {'brLen': [0, 0], 'allBrLens': [0, 0], 'NNI': [0, 0], 'eSPR': [0, 0], 'kappa': [0, 0],
                         'pi': [0, 0]}
    else:
        proposal_dict = {'brLen': [0, 0, 0, 0], 'allBrLens': [0, 0, 0, 0], 'NNI': [0, 0, 0, 0], 'eSPR': [0, 0, 0, 0],
                         'kappa': [0, 0, 0, 0], 'pi': [0, 0, 0, 0]}

    max_log_likelihood_record = -999999
    max_tree_record = -1
    max_iteration_record = 0
    max_kappa_record = kappa
    max_pi_record = base_frequency

    resample_list = []
    log_Z_list = [0]
    log_Z = 0
    epislon = 0.5
    use_residual_weight_sum = False
    if self_adaptive:
        iterations=1e10
    iteration = 0

    if feature:
        try:
            os.remove("./pilot_output/features_" + dataset + "_" + mark + ".txt")
        except Exception as e:
            e = 0

    if feature:
        os.makedirs('./pilot_output',exist_ok=True)
    else:
        os.makedirs('./DASMC_output',exist_ok=True)

    if check:
        try:
            os.remove("weight_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0
        try:
            os.remove("likelihood_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0
        try:
            os.remove("acceptance_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0
        try:
            os.remove("newicktree_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0
        try:
            os.remove("kappa_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0
        try:
            os.remove("RF_check_" + dataset + "_" + mark + "_" + str(nParticles) + "*" + str(
                iterations) + ".txt")
        except Exception as e:
            e = 0

    nParticles_list = []

    if check:
        false_rej = 0
        false_acc = 0
    FP_check_count = 0

    Evolutionary_gen = random.Random()
    Pi_gen = random.Random()
    Acceptance_gen = random.Random()
    if feature and output_feature:
        Evolutionary_gen.seed(0)
        Pi_gen.seed(0)
        Acceptance_gen.seed(0)
    else:
        Evolutionary_gen.seed(random_seed)
        Pi_gen.seed(random_seed)
        Acceptance_gen.seed(random_seed)

    print('start SMC run...')
    while iteration < iterations:

        if iteration % 1000 == 0:
            gc.collect()
            print('--------iter {}--------'.format(iteration))
            print("Phi=", Phi_list[iteration])
        iteration += 1
        RF_list = []
        rho_RF_list = numpy.zeros(nParticles)
        log_rho_RF_list = np.log(rho_RF_list)
        sur_LB_list = numpy.zeros(nParticles)
        log_proposal_ratio_list = []
        if gtr:
            kappa_tuples = [tuple(k) for k in kappa_list]
            new_kappa_tuples = []
        base_frequency_tuples = [tuple(f) for f in base_frequency_list]
        new_base_frequency_tuples = []

        detected = False
        if not feature:
            if reference:
                for k in range(nParticles):
                    RF_list.append(reference_ete_consensus.robinson_foulds(newick_tree_list[k], unrooted_trees=True)[0])
                    rho_RF_list[k] = rho_x(RF_list[k])
                    sur_LB_list[k] = int(sur_log_likelihood_ratio_list[k] > K)
                    if sur_LB_list[k] == 0 and not detected and check:
                        print('detected disfavor particle!')
                        detected = True
                min_like_and_sur_like_list = np.fmin(log_likelihood_list, ML_log_likelihood_list)
                log_rho_RF_list = np.log(rho_RF_list)

            else:
                RF_list = [None for i in range(nParticles)]
                for k in range(nParticles):
                    sur_LB_list[k] = int(sur_log_likelihood_ratio_list[k] > K)
                    if sur_LB_list[k] == 0:
                        if not detected and check:
                            print('detected disfavor particle!')
                            detected = True
                min_like_and_sur_like_list = np.fmin(log_likelihood_list, ML_log_likelihood_list)
            max_min_like_and_sur_like = np.max(min_like_and_sur_like_list)
        max_log_likelihood = np.max(log_likelihood_list)

        if self_adaptive: # self-adaptively determined the next Phi

            func = lambda x: 1 - alpha \
                if x == Phi_list[iteration - 1] \
                else (np.square(np.sum(normalized_weight_list * (
                np.exp((min_like_and_sur_like_list - max_min_like_and_sur_like - log_rho_RF_list / reference_bound) * (
                            x - Phi_list[iteration - 1])))))
                      / np.sum(normalized_weight_list * np.exp(
                        (
                                    min_like_and_sur_like_list - max_min_like_and_sur_like - log_rho_RF_list / reference_bound) * 2 * (
                                    x - Phi_list[iteration - 1]))) - alpha) \
                if x <= reference_bound \
                else (np.square(np.sum(normalized_weight_list * (
                np.exp((min_like_and_sur_like_list - max_min_like_and_sur_like) * (x - Phi_list[iteration - 1])))))
                      / np.sum(normalized_weight_list * np.exp(
                        (min_like_and_sur_like_list - max_min_like_and_sur_like) * 2 * (
                                    x - Phi_list[iteration - 1]))) - alpha) \
                if Phi_list[iteration - 1] >= reference_bound \
                else (np.square(
                np.sum(normalized_weight_list * rho_RF_list ** (Phi_list[iteration - 1] / reference_bound - 1) * (
                    np.exp((min_like_and_sur_like_list - max_min_like_and_sur_like) * (x - Phi_list[iteration - 1])))))
                      / np.sum(normalized_weight_list * rho_RF_list ** (
                                2 * Phi_list[iteration - 1] / reference_bound - 2) * np.exp(
                        (min_like_and_sur_like_list - max_min_like_and_sur_like) * 2 * (
                                    x - Phi_list[iteration - 1]))) - alpha)
            try:
                test = func(1)
                detected2 = False
            except Exception as e:
                detected2 = True

            if func(1) >= 0:
                new_Phi = 1
            else:
                if func(Phi_list[iteration - 1]) * func(1) > 0:
                    # print('Error! No solve for new Phi.')
                    new_Phi = Phi_list[iteration - 1]
                else:
                    l = Phi_list[iteration - 1]
                    r = 1
                    new_Phi = (l + r) / 2
                    while abs(func(new_Phi)) > 1e-5:
                        if func(new_Phi) > 0:
                            l = new_Phi
                            if new_Phi == (r + new_Phi) / 2:
                                break
                            new_Phi = (r + new_Phi) / 2
                        else:
                            r = new_Phi
                            if new_Phi == (l + new_Phi) / 2:
                                break
                            new_Phi = (l + new_Phi) / 2
                # if Phi_list[iteration - 1] == new_Phi:
                #     print('stay same this iter...')
                #     print('sur_log_likelihood_ratio_list:', sur_log_likelihood_ratio_list)
                #     print('normalized_weight_list:', normalized_weight_list)

        else:
            new_Phi = (iteration / iterations) ** 3
        Phi_list.append(new_Phi)

        weight_sum = 0

        for k in range(nParticles):
            if iteration == 1 or feature:
                log_weight_list[k] += log_likelihood_list[k] * (Phi_list[iteration] - Phi_list[iteration - 1])
            else:
                if not reference:
                    log_weight_list[k] += ((min(ML_log_likelihood_list[k], log_likelihood_list[k])
                                            - np.log(rho_x))
                                           * (Phi_list[iteration] - Phi_list[iteration - 1]))
                else:

                    if Phi_list[iteration] <= reference_bound:
                        log_weight_list[k] += ((min(ML_log_likelihood_list[k], log_likelihood_list[k])
                                                - np.log(rho_x(RF_list[k])) * (1 / reference_bound))
                                               * (Phi_list[iteration] - Phi_list[iteration - 1]))
                    elif Phi_list[iteration - 1] >= reference_bound:
                        log_weight_list[k] += ((min(ML_log_likelihood_list[k], log_likelihood_list[k]))
                                               * (Phi_list[iteration] - Phi_list[iteration - 1]))
                    else:
                        log_weight_list[k] += (((min(ML_log_likelihood_list[k], log_likelihood_list[k]))
                                                * (Phi_list[iteration] - Phi_list[iteration - 1]))
                                               + np.log(rho_x(RF_list[k]))
                                               * (Phi_list[iteration - 1] / reference_bound - 1))

        max_log_weight = np.max(log_weight_list)

        for k in range(nParticles):
            if use_residual_weight_sum:
                weight_sum += np.exp(log_weight_list[k] - max_log_weight)
            else:
                weight_sum += np.exp(log_weight_list[k])

        if use_residual_weight_sum:
            normalized_weight_list = np.divide(np.exp(log_weight_list - max_log_weight), weight_sum)
        else:
            normalized_weight_list = np.divide(np.exp(log_weight_list), weight_sum)

        if not feature:
            features_list = []
            proposal_list = []
            evolution_proposal = []

            for duplicate_iteration in range(duplicate_iterations):
                # The parallelization version of first half part of propagation, we don't active it because it's quick enough.

                # rand_list = [Evolutionary_gen.random() for _ in range(nParticles)]
                # rand2_list = [Pi_gen.random() for _ in range(nParticles)]
                # # for k in range(nParticles):
                # data=[(rand_list[k],rand2_list[k],mcmc_list[k],kappa_list[k],base_frequency_list[k],chain_list[k],proposal_kappa_prob, proposal_pi_prob,
                #  gtr, prior_lambda, etbrPExt, eps, bps, iteration, k) for k in range(nParticles)]
                #
                # res=propose_parallel(pool,data)
                # for k,r in enumerate(res):
                #     evolution=r[0]
                #     proposal_para=r[1]
                #     log_proposal_ratio=r[2]
                #     features_need=r[3]
                #     # tuple_new_kappa=r[4]
                #     # tuple_new_base_frequency=r[5]
                #     lnPrior=r[4]
                #     lnProp=r[5]
                #     propTree=r[6]
                #     evolution_proposal.append(evolution)
                #     proposal_list.append(proposal_para)
                #
                #     log_proposal_ratio_list.append(log_proposal_ratio)
                #
                #     features_list.append(features_need)
                #     # if gtr:
                #     #     new_kappa_tuples.append(tuple_new_kappa)
                #     # new_base_frequency_tuples.append(tuple_new_base_frequency)
                #
                #     chain_list[k].propTree=propTree.dupe()
                #     chain_list[k].propTree.data=d
                #     chain_list[k].propTree.preAndPostOrderAreValid = False
                #     chain_list[k].logPriorRatio = lnPrior
                #     chain_list[k].logProposalRatio = lnProp

                for k in range(nParticles):
                    rand = Evolutionary_gen.random()
                    log_proposal_ratio = None
                    if rand > (proposal_kappa_prob + proposal_pi_prob):
                        evolution_proposal.append(0)

                        proposal = mcmc_list[k].props.chooseProposal(False)
                        proposal_list.append(proposal)

                        features = None
                        new_kappa = kappa_list[k]
                        new_base_frequency = base_frequency_list[k]

                        if proposal.name == 'allBrLens':
                            chain_list[k].proposeAllBrLens_Modified(proposal)
                            if proposal_kappa_prob > 0:
                                if not gtr:
                                    features_list.append([0 for i in range(37)])
                                else:
                                    features_list.append([0 for i in range(45)])
                            else:
                                features_list.append([0 for i in range(35)])
                        if proposal.name == 'NNI':
                            proposal.brLenPriorType = 'exponential'
                            proposal.brLenPriorLambda = prior_lambda
                            proposal.etbrPExt = etbrPExt
                            features = chain_list[k].proposeNNI_Blaise(proposal, extract_features=True)[1]

                            if proposal_kappa_prob > 0:
                                if not gtr:
                                    features['Kappa_before'] = kappa_list[k]
                                    features['Kappa_after'] = kappa_list[k]
                                    features_list.append(list(features.values())[:37])
                                else:
                                    for ti in range(5):
                                        features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                        features['Kappa' + str(ti + 1) + '_after'] = kappa_list[k][ti]
                                    features_list.append(list(features.values())[:45])
                            else:
                                features_list.append(list(features.values())[:35])

                        if proposal.name == 'eSPR':
                            proposal.brLenPriorType = 'exponential'
                            proposal.brLenPriorLambda = prior_lambda
                            proposal.etbrPExt = etbrPExt
                            features = chain_list[k].proposeESPR_Blaise_FromETBR(proposal, extract_features=True)[1]
                            if proposal_kappa_prob > 0:
                                if not gtr:
                                    features['Kappa_before'] = kappa_list[k]
                                    features['Kappa_after'] = kappa_list[k]
                                    features_list.append(list(features.values())[:37])
                                else:
                                    for ti in range(5):
                                        features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                        features['Kappa' + str(ti + 1) + '_after'] = kappa_list[k][ti]
                                    features_list.append(list(features.values())[:45])

                            else:
                                features_list.append(list(features.values())[:35])

                    else:
                        rand2 = Pi_gen.random()

                        if rand2 < (proposal_kappa_prob / (proposal_kappa_prob + proposal_pi_prob)):
                            evolution_proposal.append(1)
                            proposal_list.append('Kappa')
                            new_base_frequency = base_frequency_list[k]
                            if not gtr:
                                rand = scipy.stats.norm.rvs(loc=0, scale=0.2, size=1)[0]
                                new_kappa = kappa_list[k] * np.exp(rand)
                            else:
                                with warnings.catch_warnings(record=True) as w:
                                    warnings.simplefilter("always")
                                    new_kappa, log_proposal_ratio = dirichlet_proposal_compact(kappa_list[k], alpha=eps)

                                    if w:
                                        warn = False
                                        for warning in w:
                                            if warning.category == RuntimeWarning:
                                                warn = True
                                        # if warn:
                                        #     print('RuntimeWarning:')
                                        #     print(f'iteration={iteration},k={k}')
                                        #     print(
                                        #         f'old kappa={kappa_list[k]}, new kappa={new_kappa}, log proposal ratio={log_proposal_ratio}')
                        else:
                            proposal_list.append('Pi')
                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                                new_kappa = kappa_list[k]
                                new_base_frequency, log_proposal_ratio = dirichlet_proposal_compact(
                                    base_frequency_list[k], alpha=bps)
                            evolution_proposal.append(new_base_frequency)

                        pTree = chain_list[k].propTree
                        for n in pTree.iterNodesNoRoot():
                            n.br.oldLen = n.br.len
                            n.br.oldNode = n
                        y0 = None
                        while not y0:
                            nNum = random.choice(pTree.preOrder)
                            if nNum != var.NO_ORDER and nNum != pTree.root.nodeNum:
                                y0 = pTree.node(nNum)
                        x0 = y0.parent
                        myRan = random.random()
                        if myRan >= 0.5:
                            x0 = y0
                            y0 = x0.parent
                        features = {}
                        features['feature_mode'] = 1
                        features['Total_brLens'] = pTree.getLenMirror(c=0)
                        features['Max_brLen'] = pTree.getMaxLen()
                        features['Var_brLen'] = pTree.getVarLen()
                        try:
                            features['Prune_brLen'] = y0.br.len
                        except Exception as e:
                            features['Prune_brLen'] = 0
                        features['Regraft_brLen'] = 0
                        features['Prune_regraft_brLen_ratio'] = -1

                        features['Prune2Regraft_leafNum'] = 0
                        features['Prune2Regraft_total_brLens'] = 0
                        features['Prune2Regraft_MaxbrLen'] = 0
                        features['Prune2Regraft_VarbrLen'] = 0

                        features['Prune2Regraft_nodeNum'] = 0
                        features['Prune2Regraft_brLen'] = 0
                        features['Prune2Regraft_Max_brLen'] = 0
                        features['Prune2Regraft_Var_brLen'] = 0

                        features['Subtree1_leafNum'] = len(list(y0.iterLeaves()))
                        features['Subtree2_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))
                        features['Subtree3_leafNum'] = 0
                        features['Subtree4_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))

                        features['Subtree1_total_brLens'], features['Subtree2_total_brLens'] = pTree.getPartLen(y0)
                        features['Subtree3_total_brLens'] = 0
                        features['Subtree4_total_brLens'] = features['Subtree2_total_brLens']
                        features['Subtree1/3_brLens_ratio'] = -1

                        features['Subtree1_max_brLen'], features['Subtree2_max_brLen'] = pTree.getPartMaxLen(y0)
                        features['Subtree3_max_brLen'], features['Subtree4_max_brLen'] = 0, features[
                            'Subtree2_max_brLen']

                        features['Subtree1_var_brLens'], features[
                            'Subtree2_var_brLens'], _, br_list2 = pTree.getPartVarLen(
                            y0)
                        features['Subtree3_var_brLens'], br_list1 = 0, []
                        unique_vals, counts = np.unique(br_list1, return_counts=True)
                        freq_dict = dict(zip(unique_vals, counts))
                        mask = numpy.ones(len(br_list2), dtype=bool)
                        for i, val in enumerate(br_list2):
                            if val in freq_dict and freq_dict[val] > 0:
                                mask[i] = False
                                freq_dict[val] -= 1
                        if len(mask) == 0 or mask.max() == False:
                            res4 = 0
                        else:
                            res4 = numpy.var(br_list2[mask])
                        features['Subtree4_var_brLens'] = res4
                        features['Large2LongRisk'] = features['Subtree1_leafNum'] * features['Regraft_brLen']
                        features['Reshape_brLens'] = 0
                        features['Reshape_mul_brLens'] = 0
                        if proposal_kappa_prob > 0:
                            if not gtr:
                                features['Kappa_before'] = kappa_list[k]
                                features['Kappa_after'] = new_kappa
                                features_list.append(list(features.values())[:37])
                            else:
                                for ti in range(5):
                                    features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                    features['Kappa' + str(ti + 1) + '_after'] = new_kappa[ti]
                                features_list.append(list(features.values())[:45])
                        else:
                            features_list.append(list(features.values())[:35])
                    if gtr:
                        new_kappa_tuples.append(tuple(new_kappa))
                    new_base_frequency_tuples.append(tuple(new_base_frequency))
                    log_proposal_ratio_list.append(log_proposal_ratio)

                this_sur_log_likelihood_ratio_list = all_model.predict(np.array(features_list))
                this_sur_log_likelihood_ratio_list += delta


                rand_acceptance_list = [Acceptance_gen.random() for _ in range(nParticles)]

                data = [(rand_acceptance_list[k], evolution_proposal[k], chain_list[k], kappa_list[k],
                         base_frequency_list[k],
                         prior_lambda, features_list[-nParticles + k], Q_list[k], name_seq_dict, newick_tree_list[k],
                         gtr, gamma, check, length,
                         log_proposal_ratio_list,
                         reference, rho_x, RF_list[k], reference_ete_consensus, reference_bound, new_Phi, iteration, k,
                         this_sur_log_likelihood_ratio_list[k], log_likelihood_list[k], ML_log_likelihood_list[k],
                         proposal_list[k],
                         K, log_prior_list[k], nFP) for k in range(nParticles)]
                res = calculate_parallel(pool, data)
                for k, r in enumerate(res):
                    flag = r[0]
                    new_log_likelihood = r[1]
                    proposal_name = r[2]
                    new_newick_tree = r[3]
                    should_acc = r[4]
                    dict_no = r[5]
                    sur_log_likelihood_ratio = r[6]
                    ML_log_likelihood = r[7]
                    new_log_prior = r[8]
                    new_kappa = r[9]
                    new_base_frequency = r[10]
                    new_Q = r[11]

                    if flag:
                        chain_list[k].curTree = chain_list[k].propTree.dupe()
                        chain_list[k].curTree.data = d
                        newick_tree_list[k] = new_newick_tree
                        sur_log_likelihood_ratio_list[k] = sur_log_likelihood_ratio
                        ML_log_likelihood_list[k] = ML_log_likelihood
                        log_likelihood_list[k] = new_log_likelihood

                        if not evolution_proposal[k]:
                            log_prior_list[k] = new_log_prior
                        else:
                            if evolution_proposal[k] == 1:
                                if not gtr:
                                    kappa_list[k] = new_kappa
                                else:
                                    kappa_list[k] = new_kappa.copy()
                            else:
                                base_frequency_list[k] = new_base_frequency.copy()
                            Q_list[k] = new_Q

                        if new_log_likelihood > max_log_likelihood_record:
                            max_iteration_record = iteration
                            max_log_likelihood_record = log_likelihood_list[k]
                            max_tree_record = new_newick_tree.write(format=1)
                            max_kappa_record = kappa_list[k]
                            max_pi_record = base_frequency_list[k]
                        proposal_dict[proposal_name][0] = proposal_dict.get(proposal_name)[0] + 1
                        if check:
                            if not should_acc:
                                false_acc += 1

                    else:

                        chain_list[k].propTree = chain_list[k].curTree.dupe()
                        chain_list[k].propTree.data = d
                        if dict_no == 0:
                            FP_check_count += 1
                        else:
                            proposal_dict[proposal_name][dict_no] = proposal_dict.get(proposal_name)[dict_no] + 1

                        chain_list[k].propTree = chain_list[k].curTree.dupe()
                        chain_list[k].propTree.data = d

                        if check:
                            if should_acc:
                                false_rej += 1

        else: # pilot SMC, a short and standard ASMC

            proposal_list = []

            for duplicate_iteration in range(duplicate_iterations):

                for k in range(nParticles):
                    rand = Evolutionary_gen.random()

                    if rand > (proposal_kappa_prob + proposal_pi_prob):

                        new_kappa = kappa_list[k]
                        chain_list[k].curTree = p4.func.readAndPop(newick_tree_list[k].write(format=1))
                        chain_list[k].curTree.data = d
                        chain_list[k].propTree = chain_list[k].curTree.dupe()
                        chain_list[k].propTree.data = d

                        proposal = mcmc_list[k].props.chooseProposal(False)
                        proposal_list.append(proposal)
                        features = None

                        if proposal.name == 'allBrLens':
                            chain_list[k].proposeAllBrLens_Modified(proposal)
                        if proposal.name == 'NNI':
                            proposal.brLenPriorType = 'exponential'
                            proposal.brLenPriorLambda = prior_lambda
                            proposal.etbrPExt = etbrPExt
                            features = chain_list[k].proposeNNI_Blaise(proposal, extract_features=True)[1]
                            if proposal_kappa_prob > 0:
                                if not gtr:
                                    features['Kappa_before'] = kappa_list[k]
                                    features['Kappa_after'] = kappa_list[k]
                                else:
                                    for ti in range(5):
                                        features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                        features['Kappa' + str(ti + 1) + '_after'] = kappa_list[k][ti]
                            if features is not None:
                                s = 'NNI '
                                for one in list(features.values()):
                                    s = s + str(one) + ' '

                        if proposal.name == 'eSPR':
                            proposal.brLenPriorType = 'exponential'
                            proposal.brLenPriorLambda = prior_lambda
                            proposal.etbrPExt = etbrPExt
                            features = chain_list[k].proposeESPR_Blaise_FromETBR(proposal, extract_features=True)[1]
                            if proposal_kappa_prob > 0:
                                if not gtr:
                                    features['Kappa_before'] = kappa_list[k]
                                    features['Kappa_after'] = kappa_list[k]
                                else:
                                    for ti in range(5):
                                        features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                        features['Kappa' + str(ti + 1) + '_after'] = kappa_list[k][ti]
                            if features is not None:
                                s = 'eSPR '
                                for one in list(features.values()):
                                    s = s + str(one) + ' '

                        new_newick_tree = Tree(chain_list[k].propTree.writeNewick(toString=True, spaceAfterComma=False))

                        for node in list(new_newick_tree.traverse())[1:]:

                            if use_P_Q_dict:
                                if gtr:
                                    kappa_item = kappa_tuples[k]
                                else:
                                    kappa_item = kappa_list[k]

                                base_frequency_item = base_frequency_tuples[k]
                                P = P_Q_dict.get((node.dist, kappa_item, base_frequency_item), [None])
                                if P[0] is None:
                                    P = np.log(expm(node.dist * Q_list[k]))
                                    P_Q_dict[(node.dist, kappa_item)] = P
                                    if len(P_Q_dict) > 300:
                                        P_Q_dict.popitem(last=False)
                            else:
                                P = np.log(expm(node.dist * Q_list[k]))
                            node.add_feature('P', P)

                            if node.is_leaf():
                                node.add_feature('sequence', name_seq_dict.get(node.name))

                        result = calc_log_likelihood(new_newick_tree, length)

                        probability_among_sites = np.dot(result[0], base_frequency_list[k].T)
                        new_log_likelihood = np.sum(np.log(probability_among_sites))

                        new_kappa = kappa_list[k]
                        if gtr:
                            new_kappa_tuples.append(tuple(new_kappa))
                        new_base_frequency = base_frequency_list[k]
                        new_base_frequency_tuples.append(tuple(new_base_frequency))

                        new_log_prior = -prior_lambda * result[1]

                        log_likelihood_ratio = new_log_likelihood - log_likelihood_list[k]

                        log_prior_ratio = new_log_prior - log_prior_list[k]
                        log_proposal_ratio = chain_list[k].logProposalRatio
                        log_ratio = log_prior_ratio + log_likelihood_ratio * new_Phi + log_proposal_ratio

                        rand_acceptance = Acceptance_gen.random()

                        if features is not None:
                            s = s + str(np.log(new_Phi)) + ' ' + str(log_likelihood_ratio) + ' ' + str(
                                log_prior_ratio) + ' ' + str(log_proposal_ratio) + ' ' + str(
                                rand_acceptance) + ' ' + str(log_likelihood_list[k])
                            feature_output_list.append(s)

                        if rand_acceptance <= np.exp(log_ratio):

                            newick_tree_list[k] = new_newick_tree

                            if output_vector:
                                if log_likelihood_ratio > 0:
                                    all_vector_list_X.append(vector_list[k])
                                vector_list[k] = TreeToVector(newick_tree_list[k])
                                if log_likelihood_ratio > 0:
                                    all_vector_list_Y.append(vector_list[k])

                            log_likelihood_list[k] = new_log_likelihood
                            log_prior_list[k] = new_log_prior
                            if log_likelihood_list[k] > max_log_likelihood_record:
                                max_iteration_record = iteration
                                max_log_likelihood_record = log_likelihood_list[k]
                                max_tree_record = new_newick_tree.write(format=1)
                                max_kappa_record = kappa_list[k]
                                max_pi_record = base_frequency_list[k]
                            proposal_dict[proposal.name][0] = proposal_dict.get(proposal.name)[0] + 1
                        else:
                            proposal_dict[proposal.name][1] = proposal_dict.get(proposal.name)[1] + 1

                    else:
                        rand2 = Pi_gen.random()
                        if rand2 < (proposal_kappa_prob / (proposal_kappa_prob + proposal_pi_prob)):
                            proposal_list.append('Kappa')
                            if not gtr:
                                rand = scipy.stats.norm.rvs(loc=0, scale=0.2, size=1)[0]
                                new_kappa = kappa_list[k] * np.exp(rand)
                            else:
                                new_kappa, log_proposal_ratio = dirichlet_proposal_compact(kappa_list[k], alpha=eps)
                            new_base_frequency = base_frequency_list[k]
                        else:
                            proposal_list.append('Pi')
                            new_kappa = kappa_list[k]
                            new_base_frequency, log_proposal_ratio = dirichlet_proposal_compact(base_frequency_list[k],
                                                                                                alpha=bps)
                        if gtr:
                            new_kappa_tuples.append(tuple(new_kappa))
                        new_base_frequency_tuples.append(tuple(new_base_frequency))

                        pTree = chain_list[k].propTree
                        for n in pTree.iterNodesNoRoot():
                            n.br.oldLen = n.br.len
                            n.br.oldNode = n
                        y0 = None
                        while not y0:
                            nNum = random.choice(pTree.preOrder)
                            if nNum != var.NO_ORDER and nNum != pTree.root.nodeNum:
                                y0 = pTree.node(nNum)
                        x0 = y0.parent
                        myRan = random.random()
                        if myRan >= 0.5:
                            x0 = y0
                            y0 = x0.parent
                        features = {}
                        features['feature_mode'] = 1
                        features['Total_brLens'] = pTree.getLenMirror(c=0)
                        features['Max_brLen'] = pTree.getMaxLen()
                        features['Var_brLen'] = pTree.getVarLen()
                        try:
                            features['Prune_brLen'] = y0.br.len
                        except Exception as e:
                            features['Prune_brLen'] = 0
                        features['Regraft_brLen'] = 0
                        features['Prune_regraft_brLen_ratio'] = -1

                        features['Prune2Regraft_leafNum'] = 0
                        features['Prune2Regraft_total_brLens'] = 0
                        features['Prune2Regraft_MaxbrLen'] = 0
                        features['Prune2Regraft_VarbrLen'] = 0

                        features['Prune2Regraft_nodeNum'] = 0
                        features['Prune2Regraft_brLen'] = 0
                        features['Prune2Regraft_Max_brLen'] = 0
                        features['Prune2Regraft_Var_brLen'] = 0

                        features['Subtree1_leafNum'] = len(list(y0.iterLeaves()))
                        features['Subtree2_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))
                        features['Subtree3_leafNum'] = 0
                        features['Subtree4_leafNum'] = pTree.data.nTax - len(list(y0.iterLeaves()))

                        features['Subtree1_total_brLens'], features['Subtree2_total_brLens'] = pTree.getPartLen(y0)
                        features['Subtree3_total_brLens'] = 0
                        features['Subtree4_total_brLens'] = features['Subtree2_total_brLens']
                        features['Subtree1/3_brLens_ratio'] = -1

                        features['Subtree1_max_brLen'], features['Subtree2_max_brLen'] = pTree.getPartMaxLen(y0)
                        features['Subtree3_max_brLen'], features['Subtree4_max_brLen'] = 0, features[
                            'Subtree2_max_brLen']

                        features['Subtree1_var_brLens'], features[
                            'Subtree2_var_brLens'], _, br_list2 = pTree.getPartVarLen(
                            y0)
                        features['Subtree3_var_brLens'], br_list1 = 0, []
                        unique_vals, counts = np.unique(br_list1, return_counts=True)
                        freq_dict = dict(zip(unique_vals, counts))
                        mask = numpy.ones(len(br_list2), dtype=bool)
                        for i, val in enumerate(br_list2):
                            if val in freq_dict and freq_dict[val] > 0:
                                mask[i] = False
                                freq_dict[val] -= 1
                        if len(mask) == 0 or mask.max() == False:
                            res4 = 0
                        else:
                            res4 = numpy.var(br_list2[mask])
                        features['Subtree4_var_brLens'] = res4
                        features['Large2LongRisk'] = features['Subtree1_leafNum'] * features['Regraft_brLen']
                        features['Reshape_brLens'] = 0
                        features['Reshape_mul_brLens'] = 0
                        if proposal_kappa_prob > 0:
                            if not gtr:
                                features['Kappa_before'] = kappa_list[k]
                                features['Kappa_after'] = new_kappa
                            else:
                                for ti in range(5):
                                    features['Kappa' + str(ti + 1) + '_before'] = kappa_list[k][ti]
                                    features['Kappa' + str(ti + 1) + '_after'] = new_kappa[ti]

                        if features is not None:
                            if proposal_list[k] == 'Kappa':
                                s = 'Kappa '
                            else:
                                s = 'Pi '
                            for one in list(features.values()):
                                s = s + str(one) + ' '

                        new_Q = calculate_Q_matrix(new_kappa, new_base_frequency)
                        new_newick_tree = newick_tree_list[k]

                        for node in list(new_newick_tree.traverse())[1:]:
                            node.P = np.log(expm(node.dist * new_Q))
                            if use_P_Q_dict:
                                if gtr:
                                    kappa_item = new_kappa_tuples[k]
                                else:
                                    kappa_item = new_kappa
                                base_frequency_item = base_frequency_tuples[k]
                                P_Q_dict[(node.dist, kappa_item, base_frequency_item)] = node.P.copy()

                        result = calc_log_likelihood(new_newick_tree, length)
                        probability_among_sites = np.dot(result[0], np.array(new_base_frequency).T)
                        new_log_likelihood = np.sum(np.log(probability_among_sites))

                        if rand2 < (proposal_kappa_prob / (proposal_kappa_prob + proposal_pi_prob)):
                            if not gtr:
                                new_log_kappa_prior = np.log(gamma.pdf(new_kappa))
                                old_log_kappa_prior = np.log(gamma.pdf(kappa_list[k]))
                                log_prior_ratio = new_log_kappa_prior - old_log_kappa_prior
                                log_proposal_ratio = np.log(new_kappa) - np.log(kappa_list[k])
                            else:
                                log_prior_ratio = 0
                        else:
                            log_prior_ratio = 0

                        log_likelihood_ratio = new_log_likelihood - log_likelihood_list[k]
                        log_ratio = log_prior_ratio + log_likelihood_ratio * new_Phi + log_proposal_ratio

                        rand_acceptance = Acceptance_gen.random()
                        if features is not None:
                            s = s + str(np.log(new_Phi)) + ' ' + str(log_likelihood_ratio) + ' ' + str(
                                log_prior_ratio) + ' ' + str(log_proposal_ratio) + ' ' + str(
                                rand_acceptance) + ' ' + str(
                                log_likelihood_list[k])
                            feature_output_list.append(s)

                        if rand_acceptance <= np.exp(log_ratio):
                            newick_tree_list[k] = new_newick_tree
                            if output_vector:
                                if log_likelihood_ratio > 0:
                                    all_vector_list_X.append(vector_list[k])
                                vector_list[k] = TreeToVector(newick_tree_list[k])
                                if log_likelihood_ratio > 0:
                                    all_vector_list_Y.append(vector_list[k])
                            log_likelihood_list[k] = new_log_likelihood

                            if not gtr:
                                kappa_list[k] = new_kappa
                            else:
                                kappa_list[k] = new_kappa.copy()
                            base_frequency_list[k] = new_base_frequency.copy()
                            Q_list[k] = new_Q

                            if log_likelihood_list[k] > max_log_likelihood:
                                max_iteration_record = iteration
                                max_log_likelihood_record = log_likelihood_list[k]
                                max_tree_record = new_newick_tree.write(format=1)
                                max_kappa_record = kappa_list[k]
                                max_pi_record = base_frequency_list[k]
                            if proposal_list[k] == 'Kappa':
                                proposal_dict['kappa'][0] = proposal_dict.get('kappa')[0] + 1
                            else:
                                proposal_dict['pi'][0] = proposal_dict.get('pi')[0] + 1
                        else:
                            if proposal_list[k] == 'Kappa':
                                proposal_dict['kappa'][1] = proposal_dict.get('kappa')[1] + 1
                            else:
                                proposal_dict['pi'][1] = proposal_dict.get('pi')[1] + 1

            if len(feature_output_list) >= 100000:
                if output_feature:
                    with open("./pilot_output/features_" + dataset + "_" + mark + ".txt", 'a') as f:
                        for line in feature_output_list:
                            f.write(line + '\n')
                feature_output_list = []

        if not use_residual_weight_sum:
            weight_sum_list.append(weight_sum)

        if new_Phi == 1:
            if use_residual_weight_sum:
                log_Z += np.log(weight_sum / nParticles) + max_log_weight
            else:
                log_Z += np.log(weight_sum / nParticles)
            log_Z_list.append(log_Z)

            break
        else:
            rESS = 1 / (nParticles * np.sum(np.square(normalized_weight_list)))
            if rESS < epislon:

                print('at iteration {}, resampling happens.'.format(iteration))
                resample_list.append(iteration)

                if use_residual_weight_sum:
                    log_Z += np.log(weight_sum / nParticles) + max_log_weight
                else:
                    log_Z += np.log(weight_sum / nParticles)
                index = np.random.multinomial(n=nParticles, pvals=normalized_weight_list, size=1)[0]

                new_mcmc_list = [-1 for i in range(nParticles)]
                new_newick_tree_list = [-1 for i in range(nParticles)]
                new_chain_list = [-1 for i in range(nParticles)]
                new_log_likelihood_list = np.array([-1.0 for i in range(nParticles)])
                new_log_prior_list = [-1.0 for i in range(nParticles)]
                new_sur_log_likelihood_ratio_list = np.array([-1.0 for i in range(nParticles)])
                new_Q_list = [-1 for i in range(nParticles)]
                new_ML_log_likelihood_list = np.array([-1.0 for i in range(nParticles)])
                new_kappa_list = [-1.0 for i in range(nParticles)]
                new_pi_list = [-1.0 for i in range(nParticles)]

                i = 0
                for n in range(nParticles):
                    while index[i] == 0:
                        i = i + 1

                    new_mcmc_list[n] = mcmc_list[i]
                    new_newick_tree_list[n] = newick_tree_list[i]
                    new_chain_list[n] = p4.Chain(chain_list[i].mcmc)
                    new_chain_list[n].curTree = chain_list[i].curTree.dupe()
                    new_chain_list[n].propTree = chain_list[i].propTree.dupe()
                    new_chain_list[n].curTree.data = d
                    new_chain_list[n].propTree.data = d
                    new_log_likelihood_list[n] = log_likelihood_list[i]
                    new_log_prior_list[n] = log_prior_list[i]
                    new_sur_log_likelihood_ratio_list[n] = sur_log_likelihood_ratio_list[i]
                    new_Q_list[n] = Q_list[i]
                    new_ML_log_likelihood_list[n] = ML_log_likelihood_list[i]
                    new_kappa_list[n] = kappa_list[i]
                    new_pi_list[n] = base_frequency_list[i]

                    index[i] = index[i] - 1

                mcmc_list = new_mcmc_list
                newick_tree_list = new_newick_tree_list
                chain_list = new_chain_list
                log_likelihood_list = new_log_likelihood_list
                log_prior_list = new_log_prior_list
                sur_log_likelihood_ratio_list = new_sur_log_likelihood_ratio_list
                Q_list = new_Q_list
                ML_log_likelihood_list = new_ML_log_likelihood_list
                kappa_list = new_kappa_list
                base_frequency_list = np.array(new_pi_list)
                log_weight_list = np.array([0.0 for i in range(nParticles)])

            log_Z_list.append(log_Z)

        if weight_sum < 1e-250 and not use_residual_weight_sum:
            use_residual_weight_sum = True

        nParticles_list.append(len(np.unique(np.array(log_likelihood_list))))

    print(f'total iteration: {iteration}')

    Q_max = calculate_Q_matrix(max_kappa_record, max_pi_record)
    if feature:
        newick_path = "./pilot_output/pilot_mcmc_consensus_" + dataset + "_" + mark + "_" + str(turn) + ".newick"
    else:
        newick_path = "./DASMC_output/mcmc_consensus_" + dataset + "_" + mark + "_" + str(turn) + ".newick"
    try:
        os.remove(newick_path)
    except Exception as e:
        e = 0

    print(f'posterior samples have {len(np.unique(np.array(log_likelihood_list)))} different values')
    print('average log-likelihood in posterior samples: ', np.average(log_likelihood_list))

    if Final_Resample and not feature:
        max_log_likelihood = np.max(log_likelihood_list)
        likelihood_list = np.exp(np.subtract(log_likelihood_list, max_log_likelihood))
        final_normalized_weight_list = np.divide(likelihood_list, np.sum(likelihood_list))
        index = np.random.multinomial(n=nParticles, pvals=final_normalized_weight_list, size=1)[0]
        new_newick_tree_list = [-1 for i in range(nParticles)]
        i = 0

        for n in range(nParticles):
            while index[i] == 0:
                i = i + 1
            new_newick_tree_list[n] = newick_tree_list[i]
            index[i] = index[i] - 1
        newick_tree_list = new_newick_tree_list

    for k in range(nParticles):
        chain_list[k].curTree = p4.func.readAndPop(newick_tree_list[k].write(format=1))
        chain_list[k].curTree.data = d
        chain_list[k].curTree.writeNewick(fName=newick_path, append=True,
                                          spaceAfterComma=False)

    print('Resample {} times.'.format(len(resample_list)))

    if gtr:
        if not ga:
            max_kappa_record = np.divide(max_kappa_record, np.sum(max_kappa_record))
    base_frequency_list = np.array(base_frequency_list)
    mean_pi = np.mean(base_frequency_list, axis=0)
    variance_pi = np.var(base_frequency_list, axis=0, ddof=1)
    std_err_pi = np.sqrt(variance_pi / nParticles)
    z = 1.96
    ci_lower_pi = mean_pi - z * std_err_pi
    ci_upper_pi = mean_pi + z * std_err_pi

    kappa_list = np.array(kappa_list)
    if gtr:
        kappa_list = kappa_list / kappa_list.sum(axis=1, keepdims=True)
    mean_kappa = np.mean(kappa_list, axis=0)
    variance_kappa = np.var(kappa_list, axis=0, ddof=1)
    std_err_kappa = np.sqrt(variance_kappa / nParticles)
    ci_lower_kappa = mean_kappa - z * std_err_kappa
    ci_upper_kappa = mean_kappa + z * std_err_kappa

    tree_length_list = []
    for newick_tree in newick_tree_list:
        tree_length_list.append(calc_log_likelihood(newick_tree, length)[1])
    if not feature:
        save_posterior_para_distribution(log_likelihood_list, tree_length_list, base_frequency_list, kappa_list,
                                         dataset, mark, turn, gtr)

    Q_con = calculate_Q_matrix(mean_kappa, mean_pi)

    print('max log-likelihood record={} at iteration {} when kappa(or evolution rate)={}, base frequency={}'.format(max_log_likelihood_record,
                                                                                     max_iteration_record,
                                                                                     max_kappa_record, max_pi_record))
    print(f'max tree record: \n {max_tree_record}')
    print('final logZ=', log_Z)
    if feature:
        for proposal, sum in proposal_dict.items():
            print('proposal {}, success {}, fail {}'.format(proposal, sum[0], sum[1]))
    else:
        for proposal, sum in proposal_dict.items():
            print(
                'proposal {}, success {}, reject but calc twice {}, reject but calc once,{}, directly reject {}'.format(
                    proposal, sum[0], sum[1], sum[2], sum[3]))

    if check:
        print(f'false rejection: {false_rej}, false accept: {false_acc}')
    if nFP:
        print(f'FP check count: {FP_check_count}')
    sumt_command = '--con --biplen -i ' + newick_path + ' -n -q --informat newick'
    sumt.main(sumt_command.split())
    try:
        os.rename(newick_path[15:-7] + '.con', newick_path[:-7] + '.con')
    except Exception as e:
        e=0
    sumt_newick = ''
    with open(newick_path[:-7] + '.con', 'r') as f:
        for one in re.split(string=f.readlines()[3].split('= ')[-1], pattern=r'\[[^]]+\]'):
            sumt_newick = sumt_newick + one
    sumt_tree = Tree(sumt_newick)

    for node in list(sumt_tree.traverse())[1:]:
        node.add_feature('P', np.log(expm(node.dist * Q_max)))
        if node.is_leaf():
            node.add_feature('sequence', name_seq_dict.get(node.name))
    result = calc_log_likelihood(sumt_tree, length)
    probability_among_sites = np.dot(result[0], np.array(max_pi_record).T)
    sumt_consensus_log_likelihood_1 = np.sum(np.log(probability_among_sites))

    sumt_tree = Tree(sumt_newick)
    for node in list(sumt_tree.traverse())[1:]:
        node.add_feature('P', np.log(expm(node.dist * Q_con)))
        if node.is_leaf():
            node.add_feature('sequence', name_seq_dict.get(node.name))
    result = calc_log_likelihood(sumt_tree, length)
    probability_among_sites = np.dot(result[0], np.array(mean_pi).T)
    sumt_consensus_log_likelihood_2 = np.sum(np.log(probability_among_sites))

    if sumt_consensus_log_likelihood_1 > sumt_consensus_log_likelihood_2:
        sumt_consensus_log_likelihood = sumt_consensus_log_likelihood_1
    else:
        sumt_consensus_log_likelihood = sumt_consensus_log_likelihood_2
    print('sumt consensus log-likelihood=', sumt_consensus_log_likelihood)

    if not taxon_namespace:
        all_taxa = sorted([one.replace('_', ' ') for one in list(name_seq_dict.keys())])
        taxon_namespace = dendropy.TaxonNamespace(all_taxa)

    sumt_dendro_consensus = dendropy.Tree.get_from_path(newick_path[:-7] + '.con',
                                                        "nexus",
                                                        taxon_namespace=taxon_namespace)

    ete_sumt = Tree(sumt_dendro_consensus.as_string(schema='newick', suppress_rooting=True), format=0)
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_support = True
    try:
        ete_sumt.sort_descendants()
    except Exception as e:
        e = 0
    if not feature:
        ete_sumt.render("./DASMC_output/consensus_tree_" + dataset + "_" + mark + "_" + str(turn) + ".png", w=800,
                    tree_style=ts)


    if feature and output_feature:
        with open("./pilot_output/features_" + dataset + "_" + mark +  ".txt", 'a') as f:
            for line in feature_output_list:
                f.write(line + '\n')

    if output_vector:
        if feature:
            with open('./pilot_output/all_vectorsX_' + dataset + '_' + mark + "_" + str(turn) + '.txt', 'w') as f1:
                with open('./pilot_output/all_vectorsY_' + dataset + '_' + mark + "_" + str(turn) + '.txt', 'w') as f2:
                    f1.write(','.join(all_vector_list_X))
                    f2.write(','.join(all_vector_list_Y))
        else:
            with open('./DASMC_output/all_vectorsX_' + dataset + '_' + mark + "_" + str(turn) + '.txt', 'w') as f1:
                with open('./DASMC_output/all_vectorsY_' + dataset + '_' + mark + "_" + str(turn) + '.txt', 'w') as f2:
                    f1.write(','.join(all_vector_list_X))
                    f2.write(','.join(all_vector_list_Y))
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kappa', default=1.8)
    parser.add_argument('-a', '--alpha', default=1 - 1e-4)
    parser.add_argument('-n', '--nParticles', default=50)
    parser.add_argument('-i', '--iterations', default=20000)
    parser.add_argument('-d', '--dataset', default='primates')
    parser.add_argument('-m', '--mark', default='0')
    parser.add_argument('-f', '--feature', default=0)
    parser.add_argument('-of', '--output_feature', default=1)
    parser.add_argument('-ov', '--output_vector', default=0)
    parser.add_argument('-s', '--self_adaptive', default=0)
    parser.add_argument('-e', '--etbrPExt', default=0.6)
    parser.add_argument('-K', '--K', default=-200)
    parser.add_argument('-del', '--delta', default=0)
    parser.add_argument('-c', '--check', default=0)
    parser.add_argument('-r', '--reference', default=0)
    parser.add_argument('-R', '--pilotRF', default=0.0)
    parser.add_argument('-p', '--proposal_kappa_prob', default=0.02)
    parser.add_argument('-pp', '--proposal_pi_prob', default=0.02)
    parser.add_argument('-b', '--brLen_prob', default=0.5)
    parser.add_argument('-t', '--turn', default=0)
    parser.add_argument('-g', '--gtr', default=0)
    parser.add_argument('-ga', '--gamma', default=0)
    parser.add_argument('-eps', '--evolution_rate_proposal_sharpness', default=200)
    parser.add_argument('-bps', '--base_frequency_proposal_sharpness', default=100)
    parser.add_argument('-upq', '--use_P_Q_dict', default=1)
    parser.add_argument('-rb', '--reference_bound', default=1.0)
    parser.add_argument('-nFP', '--no_False_Positive', default=0)
    parser.add_argument('-FR', '--Final_Resample', default=0)
    parser.add_argument('-rd', '--random_seed', default=0)
    parser.add_argument('-di', '--duplicate_iterations', default=1)
    parser.add_argument('-pc', '--parallel_cores', default=1)
    args = parser.parse_args()

    kappa = float(args.kappa)
    gtr = int(args.gtr)
    base_frequency = [0.25, 0.25, 0.25, 0.25]
    prior_lambda = 10
    if gtr:
        kappa = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        kappa = kappa / kappa.sum()
    nParticles = int(args.nParticles)
    proposal_kappa_prob = float(args.proposal_kappa_prob)
    proposal_pi_prob = float(args.proposal_pi_prob)
    check = int(args.check)
    dataset = args.dataset
    mark = args.mark
    feature = int(args.feature)
    pc=int(args.parallel_cores)
    try:
        turn = int(args.turn)
        random_seed = int(args.random_seed)
        if random_seed < 0:
            random_seed = turn + random_seed
        elif random_seed == 0:
            random_seed = turn
    except Exception as e:
        turn = args.turn
        random_seed = int(args.random_seed)
        e = 0
    if int(args.output_feature) == 1:
        random_seed = 0

    p4.func.reseedCRandomizer(random_seed)
    var.gsl_rng = p4.pf.gsl_rng_get()
    p4.pf.gsl_rng_set(var.gsl_rng, random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    pool = Pool(processes=pc, maxtasksperchild=nParticles)
    try:
        SMC(kappa=kappa, alpha=float(args.alpha), base_frequency=base_frequency, prior_lambda=prior_lambda,
            etbrPExt=float(args.etbrPExt), proposal_kappa_prob=proposal_kappa_prob,
            proposal_pi_prob=proposal_pi_prob, brLen_prob=float(args.brLen_prob),
            nParticles=nParticles, iterations=int(args.iterations), dataset=dataset, mark=mark,
            feature=feature, output_feature=int(args.output_feature), output_vector=int(args.output_vector),
            self_adaptive=int(args.self_adaptive), K=float(args.K), delta=float(args.delta), check=check,
            reference=int(args.reference), pilotRF=float(args.pilotRF), turn=turn, gtr=gtr, ga=int(args.gamma),
            eps=float(args.evolution_rate_proposal_sharpness), bps=float(args.base_frequency_proposal_sharpness),
            use_P_Q_dict=int(args.use_P_Q_dict), reference_bound=float(args.reference_bound),
            nFP=int(args.no_False_Positive),
            Final_Resample=int(args.Final_Resample), duplicate_iterations=int(args.duplicate_iterations), pool=pool,pc=pc)
    finally:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()

