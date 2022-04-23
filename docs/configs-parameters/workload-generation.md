```
{
    "notes": "constant resoure usage",
    "dataset_id": [int],
    "timesteps": [int],
    "services_types": [int],
    "workloads_var" : {
        "steps_unit":[[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]],
        "max_steps":[[10, 10], [10, 10], [10, 10], [10, 10], [10, 10]]
    },
    "plot_smoothing":301, // how much the workloads plots are smoothed
    "seed":42 // seed for generating reproducible results
}
```