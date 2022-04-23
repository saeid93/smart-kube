def action_pretty_print(action, env):
    if type(action) != str:
        action_formatted = action.reshape(env.num_services,
                                          env.num_nodes)
    else:
        action_formatted = action
    return action_formatted
