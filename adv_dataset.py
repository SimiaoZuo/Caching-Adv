import logging
import numpy as np
import pickle
import torch


logger = logging.getLogger(__name__)


class AdvSamples(object):
    """
    Wrapper class for all the adversarial samples.
    TODO: I don't know how to eliminate the race conditions...
    There are several attributes:
        src_samples: adversarial samples for the source sentences
        tgt_samples: adversarial samples for the target sentences
        idx_to_compute: index of sentences that we compute the adversarial samples
        idx_to_compute_set: set version of idx_to_compute
        src_neighbor: neighbor indices for all the source sentences
        tgt_neighbor: neighbor indices for all the target sentences
        src_sizes: length of source sentences
        tgt_sizes: length of target sentences
        num_samples: total number of samples

    """
    def __init__(self, args, src_sizes, tgt_sizes):
        # random seed are NOT set when we setup the adversarial samples
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.neighbor_file is None:
            logger.info('neighbor file is not set, use random neighbors')
        else:
            with open(args.neighbor_file, 'rb') as f:
                neighbor = pickle.load(f)
            neighbor['src_id'] = torch.LongTensor(neighbor['src_id'])
            neighbor['src_value'] = torch.FloatTensor(neighbor['src_value'])
            neighbor['tgt_id'] = torch.LongTensor(neighbor['tgt_id'])
            neighbor['tgt_value'] = torch.FloatTensor(neighbor['tgt_value'])
            neighbor['idx_to_compute'] = torch.LongTensor(neighbor['idx_to_compute'])
            assert args.num_neighbors <= neighbor['src_id'].size(1)

            if args.fp16:
                neighbor['src_value'] = neighbor['src_value'].half()
                neighbor['tgt_value'] = neighbor['tgt_value'].half()

        self.src_samples, self.tgt_samples = [], []
        self.src_sizes, self.tgt_sizes = src_sizes, tgt_sizes
        self.num_samples = len(self.src_sizes)

        if args.neighbor_file is None:
            self.idx_to_compute = torch.randperm(self.num_samples)[:int(self.num_samples * args.prop_neighbors)]
        else:
            self.idx_to_compute = neighbor['idx_to_compute']
        self.idx_to_compute_set = set(self.idx_to_compute.tolist())

        if args.neighbor_file is None:
            temp = np.random.choice(self.idx_to_compute, size=(self.num_samples, args.num_neighbors, 2), replace=True)
            temp = torch.tensor(temp)
            temp[self.idx_to_compute, 0, 0] = self.idx_to_compute
            temp[self.idx_to_compute, 0, 1] = self.idx_to_compute
            self.src_neighbor = temp[:, :, 0].clone()  # clone() is necessary for DDP to work
            self.tgt_neighbor = temp[:, :, 1].clone()
        else:
            self.src_neighbor = neighbor['src_id'][:, :args.num_neighbors]
            self.tgt_neighbor = neighbor['tgt_id'][:, :args.num_neighbors]

        src_total_size, tgt_total_size = 0, 0
        self._encoder_embed_dim = args.encoder_embed_dim
        self._decoder_embed_dim = args.decoder_embed_dim
        for i, (src_size, tgt_size) in enumerate(zip(self.src_sizes, self.tgt_sizes)):
            if i in self.idx_to_compute_set:
                src_total_size += src_size * self._encoder_embed_dim
                tgt_total_size += tgt_size * self._decoder_embed_dim

        self._dtype = torch.float16 if args.fp16 else torch.float32
        self._device = 'cpu'
        self.src_samples = torch.zeros(src_total_size, dtype=self._dtype, device=self._device).share_memory_()
        self.tgt_samples = torch.zeros(tgt_total_size, dtype=self._dtype, device=self._device).share_memory_()

        self._src_idx2mem = AdvSamples._idx2mem(self.idx_to_compute, self.src_sizes, self._encoder_embed_dim)
        self._tgt_idx2mem = AdvSamples._idx2mem(self.idx_to_compute, self.tgt_sizes, self._decoder_embed_dim)
        self._src_idx2size = AdvSamples._idx2size(self.idx_to_compute, self.src_sizes)
        self._tgt_idx2size = AdvSamples._idx2size(self.idx_to_compute, self.tgt_sizes)

        # for debugging purpose
        self._is_set_src = [0 for _ in range(self.num_samples)]
        self._is_set_tgt = [0 for _ in range(self.num_samples)]

        # alternatively, use the following code to construct shared memory block
        # note that the ctypes library does not have float16, only float32
        # import ctypes
        # import multiprocessing as mp
        # shared_array_base = mp.Array(ctypes.c_float, int(src_total_size))
        # self.src_samples = torch.from_numpy(np.ctypeslib.as_array(shared_array_base.get_obj()))
        # shared_array_base = mp.Array(ctypes.c_float, int(tgt_total_size))
        # self.tgt_samples = torch.from_numpy(np.ctypeslib.as_array(shared_array_base.get_obj()))

    @staticmethod
    def _idx2mem(ids, sizes, dim):
        """ Get the begin and end of the chunk of memory corresponding to the query index. """
        sizes = sizes[ids] * dim
        result = {}
        cum_size = sizes.cumsum().tolist()
        cum_size.insert(0, 0)
        for i, sample_id in enumerate(ids.tolist()):
            result[sample_id] = [cum_size[i], cum_size[i + 1]]
        return result

    @staticmethod
    def _idx2size(ids, sizes):
        sizes = sizes[ids]
        result = {}
        for i, sample_id in enumerate(ids.tolist()):
            result[sample_id] = sizes[i]
        return result

    def is_valid_idx(self, idx):
        if not isinstance(idx, int):
            raise KeyError('query index should be an integer')
        return idx in self.idx_to_compute_set

    def get_src_sample_from_neighbor(self, idx):
        if self.is_valid_idx(idx):
            return self.get_src_sample(idx)
        neighbor = self.src_neighbor[idx].tolist()
        result = 0
        for i, sample_id in enumerate(neighbor):
            result += self.get_src_sample(sample_id).sum(0, keepdim=True) / self.src_sizes[sample_id]
        result /= len(neighbor)
        return result

    def get_tgt_sample_from_neighbor(self, idx):
        if self.is_valid_idx(idx):
            return self.get_tgt_sample(idx)
        neighbor = self.tgt_neighbor[idx].tolist()
        result = 0
        for i, sample_id in enumerate(neighbor):
            result += self.get_tgt_sample(sample_id).sum(0, keepdim=True) / self.tgt_sizes[sample_id]
        result /= len(neighbor)
        return result

    def get_src_sample(self, idx):
        temp = self._src_idx2mem[idx]
        result = self.src_samples[temp[0]:temp[1]].clone()
        result = result.reshape(self._src_idx2size[idx], self._encoder_embed_dim)
        if self._is_set_src[idx] == 1:
            temp = result.sum().item()
            if abs(temp) < 1e-6:
                print('Warning, sample not updated in source, idx={}, norm={:.4e}'.format(idx, temp))
        return result

    def get_tgt_sample(self, idx):
        temp = self._tgt_idx2mem[idx]
        result = self.tgt_samples[temp[0]:temp[1]].clone()
        result = result.reshape(self._tgt_idx2size[idx], self._decoder_embed_dim)
        if self._is_set_tgt[idx] == 1:
            temp = result.sum().item()
            if abs(temp) < 1e-6:
                print('Warning, sample not updated in target, idx={}, norm={:.4e}'.format(idx, temp))
        return result

    def set_src_sample(self, idx, sample):
        if not self.is_valid_idx(idx):
            raise KeyError('invalid query index')
        set_sample = sample.detach().clone().view(-1).to(self._device)
        if set_sample.size(0) != self.src_sizes[idx] * self._encoder_embed_dim:
            raise ValueError('require consistent sample size')
        temp = self._src_idx2mem[idx]
        self.src_samples[temp[0]:temp[1]] = set_sample
        self._is_set_src[idx] = 1

    def set_tgt_sample(self, idx, sample):
        if not self.is_valid_idx(idx):
            raise KeyError('invalid query index')
        set_sample = sample.detach().clone().view(-1).to(self._device)
        if set_sample.size(0) != self.tgt_sizes[idx] * self._decoder_embed_dim:
            raise ValueError('require consistent sample size')
        temp = self._tgt_idx2mem[idx]
        self.tgt_samples[temp[0]:temp[1]] = set_sample
        self._is_set_tgt[idx] = 1

    def __getitem__(self, item):
        raise NotImplementedError('use the getter and setter methods')

    def __len__(self):
        return self.num_samples
