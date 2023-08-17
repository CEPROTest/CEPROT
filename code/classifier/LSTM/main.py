# encoding=utf-8
import fire

import two_stage
from eval import eval_from_config
from infer import infer_from_config
from train import train_from_config

OCD_CFG = "configs/OCD.yml"
OCD_DIR = "OCD"



def run_ocd(cfg=OCD_CFG,
            log_dir=OCD_DIR):
    train_from_config(cfg, log_dir)
    infer_from_config(cfg, log_dir)
    result = eval_from_config(cfg, log_dir)
    print(result)


def run_cup(cfg=CUP_CFG,
            log_dir=CUP_DIR):
    train_from_config(cfg, log_dir)
    infer_from_config(cfg, log_dir)
    result, _ = eval_from_config(cfg, log_dir)
    print(result)


def run_cup2(ocd_cfg=OCD_CFG,
             ocd_dir=OCD_DIR,
             cup_cfg=CUP_CFG,
             cup_dir=CUP_DIR,
             com_cfg=COM_CFG,
             com_dir=COM_DIR):
    two_stage.infer(clf_config=ocd_cfg,
                    clf_log_dir=ocd_dir,
                    upd_config=cup_cfg,
                    upd_log_dir=cup_dir,
                    com_config=com_cfg,
                    com_log_dir=com_dir)
    result = two_stage.eval(clf_config=ocd_cfg,
                            clf_log_dir=ocd_dir,
                            upd_config=cup_cfg,
                            upd_log_dir=cup_dir,
                            com_config=com_cfg,
                            com_log_dir=com_dir)
    print(result)


if __name__ == '__main__':
    fire.Fire({
        "run_ocd": run_ocd,
        "run_cup": run_cup,
        "run_cup2": run_cup2
    })