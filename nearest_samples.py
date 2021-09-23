#!/usr/bin/env python3 -u
"""
Find and cache each sample's neighbors.
"""

import logging
import numpy as np
import os
import pickle
import sys

import torch
import torch.utils.data

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from tqdm import tqdm


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    assert torch.cuda.device_count() < 2, "Use CPU or 1 GPU"
    device = 'cuda' if use_cuda else 'cpu'

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    )
    progress = progress_bar.progress_bar(
        epoch_itr.next_epoch_itr(shuffle=False),
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    assert src_dict.pad() == tgt_dict.pad() and len(models) == 1

    num_samples = task.datasets['train'].src_sizes.shape[0]  # number of samples
    n_largest = args.n_largest  # number of neighbors
    padding_idx = tgt_dict.pad()  # padding index
    model = models[0]
    model.eval()
    dim = model.args.encoder_embed_dim  # embedding dimension, i.e., KNN dimension
    # indices of samples in the neighbor bucket, i.e., neighbors are all from this bucket
    idx_to_compute = np.random.choice(np.arange(num_samples), replace=False, size=int(args.prop * num_samples))
    idx_to_compute = torch.tensor(idx_to_compute, dtype=torch.long)

    # using randperm causes program to stuck on large dataset
    # idx_to_compute = torch.randperm(num_samples)[:int(args.prop * num_samples)]

    # sentence feature is the mean of word embeddings
    # may not be able to save this to GPU for large datasets
    src_features = torch.zeros([num_samples, dim])
    tgt_features = torch.zeros([num_samples, dim])
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        target = sample['target']
        source = sample['net_input']['src_tokens']
        with torch.no_grad():
            temp = model.encoder.embed_tokens(source).sum(1) / source.ne(padding_idx).sum(1, keepdim=True)
            src_features[sample['id']] = temp.to('cpu')
            temp = model.decoder.embed_tokens(target).sum(1) / target.ne(padding_idx).sum(1, keepdim=True)
            tgt_features[sample['id']] = temp.to('cpu')
    src_neighbor_features = src_features[idx_to_compute]
    tgt_neighbor_features = tgt_features[idx_to_compute]

    # indices and cosine similarity
    src_cache = [torch.zeros([num_samples, n_largest], dtype=torch.long), torch.zeros([num_samples, n_largest])]
    tgt_cache = [torch.zeros([num_samples, n_largest], dtype=torch.long), torch.zeros([num_samples, n_largest])]
    idx_to_compute = idx_to_compute.to(device)
    src_neighbor_features = src_neighbor_features.to(device)
    tgt_neighbor_features = tgt_neighbor_features.to(device)
    src_neighbor_norm = torch.norm(src_neighbor_features, dim=-1, p=2, keepdim=True)
    tgt_neighbor_norm = torch.norm(tgt_neighbor_features, dim=-1, p=2, keepdim=True)

    @torch.no_grad()
    def cache_sim(feature, neighbor_feature, neighbor_norm, ids, cache, include_negative=False):
        norm = torch.norm(feature, dim=-1, p=2, keepdim=True)
        sim = feature @ neighbor_feature.t() / (norm @ neighbor_norm.t())

        values, indices = torch.topk(sim, k=n_largest, dim=1, sorted=True)
        temp = idx_to_compute[indices.view(-1)].view([*indices.shape])
        cache[0][ids] = temp.to('cpu')
        cache[1][ids] = values.to('cpu')

    dataset = FeatureDataset(src_features, tgt_features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.max_sentences, shuffle=False)
    for sample in tqdm(loader):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        cache_sim(sample['src'], src_neighbor_features, src_neighbor_norm, sample['id'], src_cache)
        cache_sim(sample['tgt'], tgt_neighbor_features, tgt_neighbor_norm, sample['id'], tgt_cache)

    logger.info('saving neighboring samples to {}'.format(args.save_name))
    cache = {
        'src_id': src_cache[0].tolist(),
        'src_value': src_cache[1].tolist(),
        'tgt_id': tgt_cache[0].tolist(),
        'tgt_value': tgt_cache[1].tolist(),
        'idx_to_compute': idx_to_compute.tolist(),
    }
    with open(args.save_name, 'wb') as f:
        pickle.dump(cache, f)
    logger.info('file saved')
    return None


class FeatureDataset(torch.utils.data.TensorDataset):
    """
    A wrapper class that also returns the query indices.
    """
    def __init__(self, *tensors):
        super(FeatureDataset, self).__init__(*tensors)

    def __getitem__(self, index):
        sample = {
            'src': self.tensors[0][index],
            'tgt': self.tensors[1][index],
            'id': index,
        }
        return sample


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--n-largest', type=int, default=10,
                        help='number of neighboring samples')
    parser.add_argument('--save-name', type=str,
                        help='name to save file')
    parser.add_argument('--prop', type=float, default=0.1,
                        help='proportion to compute adversarial samples')
    args = options.parse_args_and_arch(parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)


if __name__ == '__main__':
    cli_main()
