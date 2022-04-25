```
{
    "notes":[str] // "some note about the cluster",
    "nums": {
        "nodes":[int], //number of nodes in the simulation
        "services" [int], //number of services in the simulations
        "resources":[int], // number of resources in the simulations
        "services_types": [int], // number of service types in the simulations
        "services_types_map":[[int], [int], ...] // number of each contaienr types, len==services_types, sum(services_types_map)
    },
    "metrics": { // tag for each of the used matrices
        "ram":"mb",
        "cpu":"core"
    },
    "nodes_cap_rng": { // nodes capacities minimum, maximum and step
        "ram": {
            "min": [int]],
            "max": [int]],
            "step": [int]]
        },
        "cpu": { // nodes capacities minimum, maximum and step
            "min": [int],
            "max": [int],
            "step": [int]
        }
    },
    "services_request_rng": { // nodes capacities minimum, maximum and step
        "ram": {
            "min": [int],
            "max": [int],
            "step": [int]
        },
        "cpu": {
            "min": [int],
            "max": [int],
            "step": [int]
        }
    },
    "start_workload":[[float, float, ...], [float, float, ...], ...],
    "seed": 42 // seed for generating reproducible results
}
```