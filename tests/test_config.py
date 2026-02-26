from geolgm.config import load_config


def test_config_load():
    cfg = load_config("configs/base.yaml").config
    assert cfg.model.name in {"resnet18", "simple_cnn"}
    assert cfg.train.batch_size > 0
