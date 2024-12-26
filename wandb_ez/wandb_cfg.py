__all__ = ["wandb_cfg"]

wandb_cfg = {
    "key": "3914394dc58eb9d88ed682d03779576f35627195",
    "project": "IRIS_MetaNet",
    "sweep": True,
    "sweep_param": ["bit_width_list"],
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 5,
    "sweep_id": None
}