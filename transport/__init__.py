from .transport import Transport, ModelType, WeightType, PathType, Sampler

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    # --- Couette path parameters ---
    couette_mode="time",
    couette_eta_max=3.0,
    couette_nu=1.0,
    couette_tau_max=None,
    couette_alpha_min=1e-4,
    couette_freq_axes=(-2, -1),
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    - couette_*: parameters forwarded to CouettePath when path_type=="Couette"
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
        "Couette": PathType.COUETTE,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps_new = 1e-5 if train_eps is None else train_eps
        sample_eps_new = 1e-3 if train_eps is None else sample_eps
        train_eps, sample_eps = train_eps_new, sample_eps_new
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps_new = 1e-3 if train_eps is None else train_eps
        sample_eps_new = 1e-3 if train_eps is None else sample_eps
        train_eps, sample_eps = train_eps_new, sample_eps_new
    elif (path_type == PathType.COUETTE):
        # Couette has a sigma'(t) singularity at the data endpoint (same
        # class as VP / GVP); clamp by default.
        default = 1e-5 if model_type == ModelType.VELOCITY else 1e-3
        train_eps = default if train_eps is None else train_eps
        sample_eps = default if sample_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # Build path sampler (Couette is constructed directly; the other paths
    # are still picked from the legacy ``path_options`` table inside
    # ``Transport.__init__``).
    path_sampler = None
    if path_type == PathType.COUETTE:
        from .couette import CouettePath
        path_sampler = CouettePath(
            mode=couette_mode,
            eta_max=couette_eta_max,
            nu=couette_nu,
            tau_max=couette_tau_max,
            alpha_min=couette_alpha_min,
            freq_axes=tuple(couette_freq_axes),
        )

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        path_sampler=path_sampler,
    )

    return state
