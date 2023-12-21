# traffic_forecasting

## Project structure
- `download_data.sh` downloads all data files necessary to the project.
- `prepare_data.sh` unzips and does preliminary preparations to setup the data.
- `eda.ipynb` explanatory data analysis.
- `cluster_data.py` script to cluster the data.
- `learn_edge_counts.py` script to learn edges counts based on the cluster centers.

### data setup
After downloading and preparing the data using the two scripts, following directories will be created.

- **`aggregated/`**: Contains small data
- **`data/`**: Contains full/large data
- **`edge_info.csv`**: graph of the network could be constructed based on
- **`tartu_edge.csv`**: ???
