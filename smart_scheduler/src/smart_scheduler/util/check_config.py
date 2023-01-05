from typing import List
from typing import Dict, Any
from contextlib import suppress

def check_config(config: Dict[str, Any]):
    """
    check the structure of env_config_base_check
    """
    # check the for illegal items
    allowed_items = ['obs_elements', 'penalty_illegal', 'penalty_u',
                     'penalty_c', 'penalty_cv', 'penalty_v',
                     'penalty_g', 'penalty_p', 'reward_var_illegal_1',
                     'reward_var_u_1', 'reward_var_c_1', 'reward_var_cv_1',
                     'reward_var_v_1', 'reward_var_g_1', 'reward_var_p_1',
                     'reward_var_illegal_2', 'reward_var_u_2', 'reward_var_c_2',
                     'reward_var_cv_2', 'reward_var_v_2', 'reward_var_g_2',
                     'reward_var_p_2', 'episode_length', 'compute_greedy_num_consolidated',
                     'seed', 'cluster', 'workload', 'nodes_cap_rng',
                     'services_request_rng', 'num_users', 'num_stations',
                     'network', 'normalise_latency', 'trace',
                     'from_cluster', 'edge_simulator_config', 'action_method',
                     'step_method', 'kube', 'cluster_path',
                     'workload_path', 'no_action_on_overloaded', 'reward_var_one',
                     'reward_var_two', 'reward_var_three', 'reward_var_four',
                     'placement_reset', 'discrete_actions', 'max_services_nodes',
                     'backlog_size', 'job_arrival', 'target_utilization',
                     'reward_option']

    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the environment config")
    # type checks
    ints = ['episode_length', 'max_services_nodes', 'backlog_size', 'seed']
    for item in ints:
        assert type(config[item]) == int, f"<{item}> must be an integer"

    floats = ['penalty_illegal', 'penalty_u', 'penalty_c', 'penalty_v',
              'penalty_g', 'penalty_p', 'reward_var_illegal_1', 'reward_var_u_1',
              'reward_var_c_1', 'reward_var_v_1', 'reward_var_g_1', 'reward_var_p_1',
              'reward_var_illegal_2', 'reward_var_u_2',
              'reward_var_c_2', 'reward_var_v_2', 'reward_var_g_2', 'reward_var_p_2']
    for item in floats:
        assert type(config[item])==float or type(config[item])==int,\
            f"[{item}] must be a float"

    lists = ['obs_elements']
    for item in lists:
        assert type(config[item]) == list, f"<{item}> must be a list"

    strs = ['cluster_path', 'workload_path', 'network_path', 'trace_path']
    for item in strs:
        with suppress(KeyError):
            assert type(config[item]) == str, f"<{item}> must be a string"

    assert config['reward_option'] in ['rlsk', 'proposed'],\
        f"wrong input for the reward option: {config['reward_option']}"

    # observation checks
    all_obs_elements: List[str] = [
        "num_services_nodes",
        "nodes_capacities",
        "nodes_usages",
        "nodes_requests",
        "nodes_available",
        "nodes_unused",
        "nodes_slack",
        "nodes_usages_frac",
        "nodes_requests_frac",
        "num_consolidated",
        "nodes_requests_available_frac",
        "nodes_resources_unused_frac",
        "nodes_requests_available_frac_avg",
        "nodes_resources_unused_avg",
        "backlog_services_requests",
        "backlog_services_requests_frac"]

    assert set(config['obs_elements']).issubset(
        set(all_obs_elements)), f"wrong input for the obs_element <{config['obs_elements']}>"

    # observation checks
    kube: List[str] = ["admin_config",
                       "service_image",
                       "namespace",
                       "clean_after_exit",
                       "services_nodes",
                       "utilization_image",
                       "workload_path",
                       "cluster_path"]

    assert set(config['kube']).issubset(
        set(kube)), "wrong input for the kube"
    
    # fixed job_arrival 
    job_arrival_fixed: List[str] = [
        "mode",
        "interval"]
    # fixed job_arrival 
    job_arrival_bernoulli: List[str] = [
        "mode",
        "probability"]

    assert set(config['job_arrival']).issubset(set(job_arrival_fixed)) or\
        set(config['job_arrival']).issubset(set(job_arrival_bernoulli)), \
            "wrong input for the job_arrival"
