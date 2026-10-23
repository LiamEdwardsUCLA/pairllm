"""
Prompt templates for LD50, Caco-2, and Solubility: a pairwise-margin prompt
and a pointwise digit-only prompt per dataset, each presenting the compound
as SMILES + drug name.
"""

import json


def load_normalization(metadata_file):
    """Load pre-computed normalization params from a JSON file (e.g. pc3_normalization.json).
    Preferred over compute_additive_baseline when the metadata file exists — guarantees the
    same min/max used to define labels are used everywhere, even if the raw data is filtered."""
    with open(metadata_file) as f:
        return json.load(f)


## LD50 PROMPTS

LD50_PROMPT_DIGIT_ONLY = '''You are an expert toxicologist predicting the acute oral toxicity of a chemical compound. Your prediction should be given on a scale of 000 to 1000, where 000 represents the least acutely toxic compounds (highest LD50, lowest lethality) and 1000 represents the most acutely toxic (lowest LD50, highest lethality).

The toxicity metric is pLD50 = log10(1/LD50) where LD50 is in mol/kg (rat, oral). Higher pLD50 = lower LD50 = more toxic = higher score.

Compound:
- Name: {drug_name}
- SMILES: {smiles}

Base your prediction on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with your predicted label as an integer from 000 to 1000.
'''


LD50_PAIRWISE_MARGIN_PROMPT = '''You are an expert toxicologist. Below are two chemical compounds. Estimate the difference in acute oral toxicity between them on a 0–1000 scale, where 0 = least toxic (highest LD50) and 1000 = most toxic (lowest LD50).

Compound A:
- Name: {drug_name_a}
- SMILES: {smiles_a}

Compound B:
- Name: {drug_name_b}
- SMILES: {smiles_b}

Base your judgment on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with a signed integer: the toxicity score of B minus A on the 0–1000 scale.
A positive value means B is more toxic (higher score). A negative value means A is more toxic (higher score).
'''


## LD50 REPRESENTATION-ABLATION PROMPTS (Figure 3C; Figure S2A)
#
# Identical to the two standard LD50 prompts above except that the compound
# block carries a single identifier instead of both. Scale, sign convention,
# and reasoning guidance are unchanged, so a model trained under one
# representation is evaluated in-distribution.

LD50_PAIRWISE_MARGIN_PROMPT_NAME_ONLY = '''You are an expert toxicologist. Below are two chemical compounds. Estimate the difference in acute oral toxicity between them on a 0–1000 scale, where 0 = least toxic (highest LD50) and 1000 = most toxic (lowest LD50).

Compound A:
- Name: {drug_name_a}

Compound B:
- Name: {drug_name_b}

Base your judgment on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with a signed integer: the toxicity score of B minus A on the 0–1000 scale.
A positive value means B is more toxic (higher score). A negative value means A is more toxic (higher score).
'''


LD50_PAIRWISE_MARGIN_PROMPT_SMILES_ONLY = '''You are an expert toxicologist. Below are two chemical compounds. Estimate the difference in acute oral toxicity between them on a 0–1000 scale, where 0 = least toxic (highest LD50) and 1000 = most toxic (lowest LD50).

Compound A:
- SMILES: {smiles_a}

Compound B:
- SMILES: {smiles_b}

Base your judgment on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with a signed integer: the toxicity score of B minus A on the 0–1000 scale.
A positive value means B is more toxic (higher score). A negative value means A is more toxic (higher score).
'''


LD50_PROMPT_DIGIT_ONLY_NAME_ONLY = '''You are an expert toxicologist predicting the acute oral toxicity of a chemical compound. Your prediction should be given on a scale of 000 to 1000, where 000 represents the least acutely toxic compounds (highest LD50, lowest lethality) and 1000 represents the most acutely toxic (lowest LD50, highest lethality).

The toxicity metric is pLD50 = log10(1/LD50) where LD50 is in mol/kg (rat, oral). Higher pLD50 = lower LD50 = more toxic = higher score.

Compound:
- Name: {drug_name}

Base your prediction on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with your predicted label as an integer from 000 to 1000.
'''


LD50_PROMPT_DIGIT_ONLY_SMILES_ONLY = '''You are an expert toxicologist predicting the acute oral toxicity of a chemical compound. Your prediction should be given on a scale of 000 to 1000, where 000 represents the least acutely toxic compounds (highest LD50, lowest lethality) and 1000 represents the most acutely toxic (lowest LD50, highest lethality).

The toxicity metric is pLD50 = log10(1/LD50) where LD50 is in mol/kg (rat, oral). Higher pLD50 = lower LD50 = more toxic = higher score.

Compound:
- SMILES: {smiles}

Base your prediction on:
- Functional groups known to confer toxicity (e.g. heavy metals, organophosphates, reactive electrophiles, nitro/nitroso groups, halogens)
- Structural features that influence absorption, distribution, and metabolism
- Molecular size and lipophilicity as modulators of systemic exposure

Respond ONLY with your predicted label as an integer from 000 to 1000.
'''


## CACO-2 PROMPTS

CACO2_PAIRWISE_MARGIN_PROMPT = '''You are an expert pharmacokineticist. Below are two chemical compounds. Estimate the difference in Caco-2 cell permeability between them on a 0–1000 scale, where 0 = least permeable (lowest log Pcaco2) and 1000 = most permeable (highest log Pcaco2).

Compound A:
- Name: {drug_name_a}
- SMILES: {smiles_a}

Compound B:
- Name: {drug_name_b}
- SMILES: {smiles_b}

Base your judgment on:
- Lipophilicity and molecular size as primary drivers of passive permeability
- Polar surface area and hydrogen bond donor/acceptor count (high TPSA/HBD reduces permeability)
- Structural features that flag P-gp efflux substrates (large, flexible, basic nitrogen-containing molecules)
- Ionization state at physiological pH

Respond ONLY with a signed integer: the permeability score of B minus A on the 0–1000 scale.
A positive value means B is more permeable (higher score). A negative value means A is more permeable.
'''


CACO2_PROMPT_DIGIT_ONLY = '''You are an expert pharmacokineticist predicting the Caco-2 cell permeability of a chemical compound. Your prediction should be given on a scale of 000 to 1000, where 000 represents the least permeable compounds (lowest log Pcaco2) and 1000 represents the most permeable (highest log Pcaco2).

The permeability metric is log Pcaco2 in cm/s. Higher values indicate better intestinal absorption.

Compound:
- Name: {drug_name}
- SMILES: {smiles}

Base your prediction on:
- Lipophilicity and molecular size as primary drivers of passive permeability
- Polar surface area and hydrogen bond donor/acceptor count (high TPSA/HBD reduces permeability)
- Structural features that flag P-gp efflux substrates (large, flexible, basic nitrogen-containing molecules)
- Ionization state at physiological pH

Respond ONLY with your predicted label as an integer from 000 to 1000.
'''


## SOLUBILITY PROMPTS

SOLUBILITY_PAIRWISE_MARGIN_PROMPT = '''You are an expert medicinal chemist. Below are two chemical compounds. Estimate the difference in aqueous solubility between them on a 0–1000 scale, where 0 = least soluble (lowest log S) and 1000 = most soluble (highest log S).

Compound A:
- Name: {drug_name_a}
- SMILES: {smiles_a}

Compound B:
- Name: {drug_name_b}
- SMILES: {smiles_b}

Base your judgment on:
- Hydrogen bond donors and acceptors (increase solubility via hydration)
- Ionizable groups and salt forms (dramatically increase aqueous solubility)
- Lipophilicity and aromaticity (flat, lipophilic molecules have poor solubility due to crystal packing)
- Molecular complexity and rotatable bonds (flexible molecules disrupt crystalline order)

Respond ONLY with a signed integer: the solubility score of B minus A on the 0–1000 scale.
A positive value means B is more soluble (higher score). A negative value means A is more soluble.
'''


SOLUBILITY_PROMPT_DIGIT_ONLY = '''You are an expert medicinal chemist predicting the aqueous solubility of a chemical compound. Your prediction should be given on a scale of 000 to 1000, where 000 represents the least soluble compounds (lowest log S) and 1000 represents the most soluble (highest log S).

The solubility metric is log S in mol/L. Higher values indicate greater aqueous solubility.

Compound:
- Name: {drug_name}
- SMILES: {smiles}

Base your prediction on:
- Hydrogen bond donors and acceptors (increase solubility via hydration)
- Ionizable groups and salt forms (dramatically increase aqueous solubility)
- Lipophilicity and aromaticity (flat, lipophilic molecules have poor solubility due to crystal packing)
- Molecular size and flexibility

Respond ONLY with your predicted label as an integer from 000 to 1000.
'''

