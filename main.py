from trainers.train import Trainer

import argparse
import time
parser = argparse.ArgumentParser()
import pandas

if __name__ == "__main__":
    start_time = time.time()

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    #parser.add_argument('--save_dir',               default='logs/docker_final_repro2/CNN',         type=str, help='Directory containing all experiments')
    parser.add_argument('--save_dir', default='logs/clean', type=str, help='Directory containing all experiments')

    parser.add_argument('--exp_name', default='EXP1', type=str, help='experiment name')
    #parser.add_argument('--exp_name',               default='NoParam_SRC2_FNO',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='UniJDOT',               type=str, help='UDA (UAN), OVANet, DANCE, PPOT, UniOT, UniJDOT')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default='./data',            type=str, help='Path containing dataset')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='Mantis',                      type=str, help='Backbone of choice: (CNN - FNO - S3Layer - TSLANet - Mantis - Moment - TimesNet - Chronos - PatchTST - Mamba - MambaFast - MambaSSM)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=10,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default="cuda",                   type=str, help='cpu or cuda')
    parser.add_argument("--uniDA",                  action='store_false', help='Different Label Set between Src and Trg Domain ?')
    parser.add_argument("--generate-private", action='store_false', help='uniDA should be True too ?')
    parser.add_argument("--log_wandb", action='store_true', help='log results using wandb')
    parser.add_argument('--wandb_project', default=None, type=str, help='W&B project name')
    parser.add_argument('--wandb_entity', default='myteam', type=str, help='W&B entity/team name')
    parser.add_argument('--hparams_json', default=None, type=str,
                        help='Path to a JSON hparams file (e.g. configs/best_hparams/best_hparams_CNN.json). '
                             'When set, overrides the hparams_*.py configs for the selected dataset/method.')
    parser.add_argument('--auto_threshold', action='store_true',
                        help='Enable automatic threshold selection (overrides auto_threshold hparam)')
    parser.add_argument('--threshold_method', default=None,
                        choices=['yen', 'otsu', 'li', 'triangle'],
                        help='Auto-thresholding method when --auto_threshold is set (default: yen)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume run from existing results_temp.csv, skipping already completed (scenario, run) pairs')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # Inject CLI threshold overrides into hparams after trainer init
    if args.auto_threshold:
        trainer.hparams['auto_threshold'] = True
    if args.threshold_method is not None:
        trainer.hparams['threshold_method'] = args.threshold_method

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()

    gap = (time.time() - start_time)
    print("--- %s seconds ---" % (gap))