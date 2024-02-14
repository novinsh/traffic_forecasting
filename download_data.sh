#!/bin/bash
# we store the data files on dropbox. this script downloads them to the root of the project.
# data
wget "https://www.dropbox.com/scl/fi/nzo0r46xf1px5r37n96h5/aggregated.zip?rlkey=hhehoy72utuz8s1j7dfhag66x&dl=0" -O data_small.zip
wget "https://www.dropbox.com/scl/fi/vjch1875tn6qyrxglpmy0/data.zip?rlkey=wt18fzfj8054v4q219eva4xvd&dl=0" -O data_large.zip

# edge info
wget "https://www.dropbox.com/scl/fi/4thhh8yjhh8zagtpxb2cr/edgeinfo.csv?rlkey=wxb87kbvzwvn6s74r89q5ay41&dl=0" -O edge_info.csv
wget "https://www.dropbox.com/scl/fi/itb2iv38cqhnf74cz7zv0/tartu_edge.csv?rlkey=gmycfuiutcv0xcy5yeka6wq6c&dl=0" -O tartu_edge.csv

# saved models
# download saved models to reuse
wget "https://www.dropbox.com/scl/fi/f3y2xj662yfam8de63zu5/cluster_model.pkl?rlkey=msfffwfam06d41ruw0besa4ox&dl=0" -O cluster_model.pkl
wget "https://www.dropbox.com/scl/fi/didmledouri7ibnrnsoag/cluster_model_medoids_ratio_.zip?rlkey=7yx41wwtjmi0xago3kvwjko2v&dl=0" -o cluster_model_medoids_ratio_.zip
