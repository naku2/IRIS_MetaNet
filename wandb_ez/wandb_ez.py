import wandb
from configs.config import wandb_cfg, layer_cfg
import os.path
# from .wandb_cfg import *

api = None

def init(args, train_func):
    global api
    wandb.login(key=wandb_cfg.key, relogin = True)
    if wandb_cfg.resume:
        resume_string = wandb_cfg.resume
        entity, project, run_id = resume_string.split("/")
        api = wandb.Api(overrides={"entity": entity, "project": project})
    else:
        api = wandb.Api(overrides={"entity": wandb_cfg.entity, "project": wandb_cfg.project})
    if wandb_cfg.sweep_enabled:
        init_sweep(args, train_func)
    else:
        init_once(args)
        train_func()
    return


def init_once(args):
    config = args2config_once(args)
    if wandb_cfg.resume:
        resume_string = wandb_cfg.resume
        entity, project, run_id = resume_string.split("/")
        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id, 
            resume="must"
        )
    elif wandb_cfg.name != None:
        run = wandb.init(
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            config=config
        )
    else:
        run = wandb.init(
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
            config=config
        )
    return 


def init_sweep(args, train_func):
    if wandb_cfg.sweep_id:
        wandb.agent(wandb_cfg.sweep_id, train_func, count=wandb_cfg.sweep_count)
        return 
    
    config = args2config_sweep(args)
    sweep_config = wandb_cfg.sweep_config
    print(wandb_cfg)
    sweep_config["parameters"] = config
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=wandb_cfg.entity, 
        project=wandb_cfg.project)
    wandb.agent(sweep_id, train_func, count=wandb_cfg.sweep_count)
    return 

def nested_dict_sweep(args):
    if not isinstance(args, dict):
        return {"values": args if isinstance(args, list) else [args]}
    
    ret = {}
    for key, value in args.items():
        ret[key] = nested_dict_sweep(value)
    
    return {"parameters": ret}


def args2config_once(args):
    # convert argparse into dict
    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]
    return config


def args2config_sweep(args):
    # config = args2config_once(args)
    config = {}
    for key in args.__dict__:
        config[key] = {"values": args.__dict__[key] if type(args.__dict__[key]) is list else [args.__dict__[key]]}

    if layer_cfg is not None:
        config["layer_cfg"] = nested_dict_sweep(layer_cfg)
    # list_keys = wandb_cfg["sweep_param"]
    # for key in list_keys:
    #     config[key] = {"values": list(map(int, args.__dict__[key].split(',')))}
    #     print(config[key])

    return config


def log(report_dict, model=None, weight_distributions=None):
    """
    Logs weight distributions to WandB.
    Parameters:
        report_dict: General metrics to log
        model: The PyTorch model (required if logging weights)
        weight_distributions: Precomputed weight distributions
        step: The step to log (default: 0 for limiting to the first step)
    """
    if model is not None:
        print("Logging weight distributions...")
        weight_distributions = {"quantized_weights": {}, "raw_weights": {}}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log raw weights
                weight_distributions["raw_weights"][f"weights/{name}"] = wandb.Histogram(param.cpu().detach().numpy())
                # Check if quantization is applied
                if hasattr(param, "quantize_fn"):
                    quantized_weight = param.quantize_fn(param).cpu().detach().numpy()
                    weight_distributions["quantized_weights"][f"quantized_weights/{name}"] = wandb.Histogram(quantized_weight)

        # Update the report_dict
        report_dict.update(weight_distributions["raw_weights"])
        report_dict.update(weight_distributions["quantized_weights"])
        print(f"Weight distributions added: {list(weight_distributions['raw_weights'].keys()) + list(weight_distributions['quantized_weights'].keys())}")

    # Log to WandB with step
    wandb.log(report_dict)
    return

def upload_model(model_path):
    if os.path.isfile(model_path + '/model_best.pth.tar'):
        artifact = wandb.Artifact(wandb.run.name + 'model_best', type='model')
        artifact.add_file(model_path + '/model_best.pth.tar')
        wandb.run.log_artifact(artifact)
        artifact.wait()

    artifact = wandb.Artifact(wandb.run.name + 'model_latest', type='model')
    artifact.add_file(model_path + '/model_latest.pth.tar')
    wandb.run.log_artifact(artifact)
    artifact.wait()
    
    runs = api.run(wandb.run.id)

    for artifact in runs.logged_artifacts():
        if 'latest' not in artifact.aliases:
            artifact.delete(delete_aliases=True)

    return


def finish():
    wandb.finish()
    return