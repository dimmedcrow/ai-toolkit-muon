import torch


def _normalize_param_groups(params, default_lr):
    if isinstance(params, (list, tuple)):
        normalized_params = list(params)
    else:
        normalized_params = list(params)

    if not normalized_params:
        return []

    if isinstance(normalized_params[0], dict):
        param_groups = []
        for group in normalized_params:
            group_params = list(group.get('params', []))
            if not group_params:
                continue
            param_groups.append({
                'params': group_params,
                'lr': group.get('lr', default_lr),
            })
        return param_groups

    return [{
        'params': normalized_params,
        'lr': default_lr,
    }]


def _get_muon_optimizer(params, learning_rate, optimizer_params):
    try:
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    except ImportError:
        raise ImportError("Please install muon-optimizer to use Muon optimizer -> pip install muon-optimizer")

    param_groups = _normalize_param_groups(params, learning_rate)

    # Muon is intended for GPU training. If CUDA is available, keep all trainable
    # params on the active CUDA device to avoid accidentally running Muon on CPU.
    if torch.cuda.is_available():
        target_device = torch.device('cuda', torch.cuda.current_device())
        for group in param_groups:
            for param in group['params']:
                if hasattr(param, 'device') and param.device != target_device:
                    param.data = param.data.to(target_device)
                    if param.grad is not None:
                        param.grad = param.grad.to(target_device)

    muon_momentum = optimizer_params.get('momentum', 0.95)
    muon_lr = optimizer_params.get('muon_lr', 0.02)
    adam_betas = optimizer_params.get('betas', (0.9, 0.95))
    adam_eps = optimizer_params.get('eps', 1e-10)
    weight_decay = optimizer_params.get('weight_decay', 0.0)

    muon_param_groups = []
    for group in param_groups:
        muon_params = []
        aux_adam_params = []

        for param in group['params']:
            if getattr(param, 'ndim', 0) >= 2:
                muon_params.append(param)
            else:
                aux_adam_params.append(param)

        if muon_params:
            muon_param_groups.append({
                'params': muon_params,
                'lr': group['lr'] if group['lr'] != learning_rate else muon_lr,
                'momentum': muon_momentum,
                'weight_decay': weight_decay,
                'use_muon': True,
            })

        if aux_adam_params:
            muon_param_groups.append({
                'params': aux_adam_params,
                'lr': group['lr'],
                'betas': adam_betas,
                'eps': adam_eps,
                'weight_decay': weight_decay,
                'use_muon': False,
            })

    if not muon_param_groups:
        raise ValueError('Muon optimizer received no parameters to optimize')

    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return MuonWithAuxAdam(muon_param_groups)
    return SingleDeviceMuonWithAuxAdam(muon_param_groups)


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type == 'muon':
        optimizer = _get_muon_optimizer(params, learning_rate, optimizer_params)
    elif lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        import bitsandbytes

        if lower_type == "adam8bit":
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        if lower_type == "ademamix8bit":
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "adamw8bit":
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "lion8bit":
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
