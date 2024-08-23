import time
import torch
from torch import nn
import args
import torch.nn.functional as F
import os
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from apex import optimizers
import subprocess
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from utils import normalize_smiles, LpLoss


# create a function (this my favorite choice)
def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()

        self.config = config
        self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer = tokenizer
        self.min_loss = {
            self.hparams.measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.measure_name + "min_epoch": 0,
        }

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
        )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        # if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []
        self.loss = LpLoss()  # mean absolute error
        # decoder
        self.net = self.Net(
            config.n_embd, dims=config.dims, dropout=config.dropout,
        )

    class Net(nn.Module):
        dims = [150, 50, 50, 2]

        def __init__(self, smiles_embed_dim, dims=dims, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True
            self.fcs = []  # nn.ModuleList()
            # print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 1)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            return z

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)

        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def get_loss(self, smiles_emb, measures):

        z_pred = self.net.forward(smiles_emb)  #.squeeze()
        measures = measures.float()
        # print(f'z_pred: {z_pred}')
        # print(f'measures: {measures}')
        # print(f'>> z_pred size: {z_pred.size()}, measures size: {measures.size()}')
        return self.loss(z_pred, measures), z_pred, measures

    def on_save_checkpoint(self, checkpoint):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state'] = torch.get_rng_state()
        out_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state'] = np.random.get_state()
        if random:
            out_dict['python_state'] = random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        # load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key == 'torch_state':
                torch.set_rng_state(value)
            elif key == 'cuda_state':
                torch.cuda.set_rng_state(value)
            elif key == 'numpy_state':
                np.random.set_state(value)
            elif key == 'python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if self.hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.99)
        # print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idx = batch[0]
        mask = batch[1]
        targets = batch[-1]

        loss = 0
        loss_tmp = 0

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('train_loss', loss, on_step=True)

        logs = {"train_loss": loss}

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        idx = val_batch[0]
        mask = val_batch[1]
        targets = val_batch[-1]

        loss = 0
        loss_tmp = 0
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual = self.get_loss(loss_input, targets)
        self.log('train_loss', loss, on_step=True)
        return {
            "val_loss": loss,
            "pred": pred.detach(),
            "actual": actual.detach(),
            "dataset_idx": dataset_idx,
        }

    def validation_epoch_end(self, outputs):
        # results_by_dataset = self.split_results_by_dataset(outputs)
        tensorboard_logs = {}
        for dataset_idx, batch_outputs in enumerate(outputs):
            dataset = self.hparams.dataset_names[dataset_idx]
            print("x_val_loss: {}".format(batch_outputs[0]['val_loss'].item()))
            avg_loss = torch.stack([x["val_loss"] for x in batch_outputs]).mean()
            preds = torch.cat([x["pred"] for x in batch_outputs])
            actuals = torch.cat([x["actual"] for x in batch_outputs])
            val_loss = self.loss(preds, actuals)

            actuals_cpu = actuals.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            pearson_r = pearsonr(actuals_cpu, preds_cpu)
            r2 = r2_score(actuals_cpu, preds_cpu)
            tensorboard_logs.update(
                {
                    # dataset + "_avg_val_loss": avg_loss,
                    self.hparams.measure_name + "_" + dataset + "_loss": val_loss,
                    self.hparams.measure_name + "_" + dataset + "_r2": r2,
                    self.hparams.measure_name + "_" + dataset + "_pearsonr": pearson_r[0],
                }
            )

        if (
                tensorboard_logs[self.hparams.measure_name + "_valid_loss"]
                < self.min_loss[self.hparams.measure_name + "min_valid_loss"]
        ):
            self.min_loss[self.hparams.measure_name + "min_valid_loss"] = tensorboard_logs[
                self.hparams.measure_name + "_valid_loss"
                ]
            self.min_loss[self.hparams.measure_name + "min_test_loss"] = tensorboard_logs[
                self.hparams.measure_name + "_test_loss"
                ]
            self.min_loss[self.hparams.measure_name + "min_epoch"] = self.current_epoch

        tensorboard_logs[self.hparams.measure_name + "_min_valid_loss"] = self.min_loss[
            self.hparams.measure_name + "min_valid_loss"
            ]
        tensorboard_logs[self.hparams.measure_name + "_min_test_loss"] = self.min_loss[
            self.hparams.measure_name + "min_test_loss"
            ]

        self.logger.log_metrics(tensorboard_logs, self.global_step)

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k])

        print("Validation: Current Epoch", self.current_epoch)
        append_to_file(
            os.path.join(self.hparams.results_dir, "results_" + ".csv"),
            f"{self.hparams.measure_name}, {self.current_epoch},"
            + f"{tensorboard_logs[self.hparams.measure_name + '_valid_loss']},"
            + f"{tensorboard_logs[self.hparams.measure_name + '_test_loss']},"
            + f"{self.min_loss[self.hparams.measure_name + 'min_epoch']},"
            + f"{self.min_loss[self.hparams.measure_name + 'min_valid_loss']},"
            + f"{self.min_loss[self.hparams.measure_name + 'min_test_loss']}",
        )

        return {"avg_val_loss": avg_loss}


def get_dataset(data_root, filename, dataset_len, aug, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df, measure_name, aug)
    return dataset


class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name, tokenizer, aug=True):
        df = df.dropna()  # TODO - Check why some rows are na
        self.df = df
        all_smiles = df["smiles"].tolist()
        self.original_smiles = []
        self.original_canonical_map = {
            smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
        }

        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        if measure_name:
            all_measures = df[measure_name].tolist()
            self.measure_map = {all_smiles[i]: all_measures[i] for i in range(len(all_smiles))}

        # Get the canonical smiles
        # Convert the keys to canonical smiles if not already

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")

        self.aug = aug
        self.is_measure_available = "measure" in df.columns

    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        canonical_smiles = self.original_canonical_map[original_smiles]
        # print(canonical_smiles, self.measure_map[original_smiles])
        return canonical_smiles, self.measure_map[original_smiles]

    def __len__(self):
        return len(self.original_smiles)


class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
                 prefix="N-Step-Checkpoint",
                 use_modelcheckpoint_filename=False,
                 ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
                # filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def collate(batch):
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    tokens = tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
    return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']),
            torch.tensor([smile[1] for smile in batch]))


def model_forward(model, batch, myloss, device):
    idx = batch[0].to(device)
    mask = batch[1].to(device)
    targets = batch[-1].to(device)

    loss = 0
    loss_tmp = 0

    b, t = idx.size()
    # print(b, t)

    # print('\n')
    # print(f'******** input is: {idx}')

    token_embeddings = model.tok_emb(idx)  # each index maps to a (learnable) vector

    # print(f'******** tokenized input is: {token_embeddings}')
    # print('\n')

    x = model.drop(token_embeddings)
    x = model.blocks(x, length_mask=LM(mask.sum(-1)))
    token_embeddings = x
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    loss_input = sum_embeddings / sum_mask
    loss, pred, actual = model.get_loss(loss_input, targets)
    # print(pred, actual)
    l2_error = myloss(pred.reshape(b, -1), actual.reshape(b, -1)).item()

    return pred, actual, l2_error


def main():
    margs = args.parse_args()
    pos_emb_type = 'rot'
    # print('pos_emb_type is {}'.format(pos_emb_type))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> GPU unavailable. Device switched to {device}')
        margs.device = device
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))
    # print(run_name)

    if type(margs) is dict:
        margs = Namespace(**margs)

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')

    predict_filename = margs.dataset_name + "_" + "predict.csv"
    test_ds = get_dataset(margs.data_root, predict_filename, margs.eval_dataset_length,
                          aug=False, measure_name=margs.measure_name)
    ntest = len(test_ds)
    # print(f'>> Test dataset size is: {ntest}')
    test_loader = DataLoader(test_ds,
                             batch_size=margs.batch_size,
                             num_workers=margs.num_workers,
                             shuffle=False,
                             collate_fn=collate)

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.measure_name)
    margs.checkpoint_root = checkpoint_root
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models")
    results_dir = os.path.join(checkpoint_root, "results_predict")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # print(margs)

    seed.seed_everything(margs.seed)

    saved_checkpoints = [checkpoint for checkpoint in os.listdir(checkpoint_dir) if checkpoint.startswith('checkpoint_' + margs.measure_name)]
    if len(saved_checkpoints) > 1:
        print('\n')
        print('-' * 100)
        print('>> WARNING: multiple checkpoints exist. Loading the first one by default..')
        print('-' * 100)
        print('\n')

    # last_checkpoint_file = os.path.join(checkpoint_dir, "best.ckpt")
    last_checkpoint_file = os.path.join(checkpoint_dir, saved_checkpoints[0])
    if os.path.isfile(last_checkpoint_file):
        print(f">> Making predictions from : {last_checkpoint_file}")
        model = LightningModule(margs, tokenizer).load_from_checkpoint(last_checkpoint_file,
                                                                       strict=True,
                                                                       config=margs,
                                                                       tokenizer=tokenizer,
                                                                       vocab=len(tokenizer.vocab)).to(device)
    else:
        print(f">> Error: cannot find pretrained model. Prediction aborted!!")
        exit()

    myloss = LpLoss(size_average=False)

    res_filename = os.path.join(margs.results_dir, "results_" + ".csv")
    with open(res_filename, "w") as f:
        f.write("prediction, gt, l2_error\n")

    pred_all = []
    gt_all = []
    model.eval()
    with torch.no_grad():
        test_l2 = 0
        tic = time.perf_counter()
        for batch in test_loader:
            # print(f'batch print: {batch}')
            out, yy, l2_error = model_forward(model, batch, myloss, device)
            # print(type(out), type(yy), type(l2_error))
            append_to_file(res_filename, f"{out.item()}, {yy.item()}, {l2_error}")
            pred_all.append(out.item())
            gt_all.append(yy.item())
            test_l2 += l2_error
        test_l2 /= ntest
        toc = time.perf_counter()

    r2 = r2_score(gt_all, pred_all)

    print(f'>> Evaluation completed. Time taken: {(toc - tic):.2f}s, relative L2 test error: {test_l2:.5f}')
    print(f'>> Predicted results: {pred_all}')


if __name__ == '__main__':
    main()
