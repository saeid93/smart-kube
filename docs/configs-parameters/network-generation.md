```
{
    "notes": <some text explaining the experiment>,
    "dataset_id": [int], // dataset that we are going to generate the network for that
    "num_users": [int], // number of users in the network
    "num_stations": [int], // number of stations in the network
    "width": [float], // width of the grid of the network
    "length": [float], // length of the grid of the network
    "speed_limit": [int], // speed_limit of the user_movements in the non-random
    "from_dataset": [bool], // from the preprocessed dataset or not
    "users_containers_distributions": [ "equal", "random"|] // connection of the users container to be random or equally scattered
    "dataset_metadata": [int], // which dataset metadata
    "nodes_stations_con": [int], // the number of connection between nodes and stations
    "seed": [int] // seed for generating reproducible results
}
```