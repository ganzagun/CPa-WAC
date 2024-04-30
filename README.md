# CPa-WAC
This repo contains code for the following paper:
1. IJCAI 2024 "CPa-WAC: Constellation Partitioning-based Scalable Weighted Aggregation Composition for Knowledge Graph Embedding".

## Preparing datasets
To run experiments for dataset used in the paper, please download all the datasets and put them under `data/`


## Usage

### Preparing Clusters
python CPa_WAC_clustering.py --dataset WN18RR --beta 200 --delta 40 --gamma 1 --sigma 18000 --cluster_folder cluster_c2


### Running Encoders 
python KGconstellation_dencoder.py --dataset WN18RR --cluster_folder cluster_c2  --cluster_num 2 



### Running Decoders
python KGconstellation_dencoder.py --dataset WN18RR --cluster_folder cluster_c2  --cluster_total 2 


## Citing CPa-WAC

If you find CPa-WAC useful, please cite our paper.
