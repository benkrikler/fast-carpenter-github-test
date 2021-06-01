# Examples

## CMS Level 1 Trigger study

```bash
git clone https://github.com/FAST-HEP/fast-carpenter.git
cd fast-carpenter

# download data
mkdir -p data
wget \
  -O data/CMS_L1T_study.root \
  http://fast-hep-data.web.cern.ch/fast-hep-data/cms/L1T/CMS_L1T_study.root

# if not already installed
pip install -e .
fast_carpenter \
  examples/cms/cms_l1t_data.yml \
  examples/cms/cms_l1t_processing.yml \
  --outdir=examples/output/cms/cmsl1t
```

## DUNE Supernova Neutrino Study

```bash
git clone https://github.com/FAST-HEP/fast-carpenter.git
cd fast-carpenter

# download data
mkdir -p data
# ask a DUNE member

# if not already installed
pip install -e .
fast_carpenter \
  examples/dune/dune_snnu_data.yml \
  examples/dune/dune_snnu_processing.yml \
  --outdir=examples/output/dune/dune_snnu
```
