#!/bin/bash

if [ "$1" = "clean" ]; then
    echo "Performing clean operation..."
    # delete downloaded files and folders
    rm aggregated -r
    rm data -r
    rm data_large.zip
    rm data_small.zip
    rm edge_info.csv
    rm tartu_edge.csv
    rm cluster_model.pkl
    rm cluster_model_medoids_ratio_.zip
else
    echo "Performing data operations..."
    # Unzip and preliminary data preparation 
    unzip data_small.zip
    unzip data_large.zip
    unzip cluster_model_medoids_ratio_.zip
fi
