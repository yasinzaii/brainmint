import sys
from pathlib import Path

import pytest

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict, OmegaConf

from brainmint.data.transforms.modality_choice import ChooseStreamForModalitiesd
from brainmint.data.transforms.stream_mapping import SampleLatentsFromKeysd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _iter_transforms(transform):
    if transform is None:
        return
    seen = set()
    stack = [transform]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue

        yield current

        children = []
        nested = getattr(current, "transform", None)
        if nested is not None and nested is not current:
            children.append(nested)
        nested = getattr(current, "_transform", None)
        if nested is not None and nested is not current:
            children.append(nested)
        transforms = getattr(current, "transforms", None)
        if transforms:
            children.extend(list(transforms))
        stack.extend(reversed(children))


@pytest.mark.parametrize(
    "config_name",
    [
        "exp/train/translation/stage_a",
        # "exp/train/translation/stage_b",
        # "exp/train/translation/stage_c",
    ],
)
def test_translation_configs_build(config_name):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            "dataset.brainscape.subset_frac=0.01",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",
        ]
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
        with open_dict(cfg):
            cfg.hydra.job.num = 0
        HydraConfig.instance().set_config(cfg)
        print(OmegaConf.to_yaml(cfg, resolve=True))

        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        trainloader = dm.train_dataloader()

        train_tf = getattr(trainloader.dataset, "transform", None)
        assert train_tf is not None
        composed_tf = getattr(train_tf, "transform", train_tf)

        choice_tf = None
        sampling_tf = None
        for transform in _iter_transforms(composed_tf):
            if isinstance(transform, ChooseStreamForModalitiesd):
                choice_tf = transform
            if isinstance(transform, SampleLatentsFromKeysd):
                sampling_tf = transform

        assert choice_tf is not None
        assert sampling_tf is not None

        choice_state = choice_tf.state
        sampling_state = sampling_tf.sampling_state

        new_choices = {
            "s_real": {
                "*": {
                    "streams": {"real": ["latent", "latent_sigma"]},
                    "probs": {"real": 1.0},
                }
            }
        }
        
        for batch in trainloader:
            pass
        
        choice_state.set_epoch(1)
        choice_state.set_choices(new_choices)
        sampling_state.set_config({"sigma_prob": 0.5, "sigma_alpha": 1.0})

        for batch in trainloader:
            pass

        batch = next(iter(trainloader))
        assert "t1w" in batch
        assert "t2w" in batch
        assert "is_mod_synthetic" in batch
        assert "modality_map" in batch
