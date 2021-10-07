def get_params_for_component(params, component):
    """
    Returns a dictionary of all params for one component defined
    in params in the form component__param: value

    e.g.
    >> params = {"vec__min_df": 1, "clf__probability": True}
    >> get_params_for_component(params, "vec")
    {"min_df": 1}
    """
    component_params = {}
    for k, v in params.items():
        if k.startswith(component):
            _, component_arg = k.split(f"{component}__")
            component_params[component_arg] = v
    return component_params
