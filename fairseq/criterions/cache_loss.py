import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

logger = logging.getLogger(__name__)


def reduce_cache_metrics(logging_outputs):
    """Aggregate logging outputs from data parallel training."""
    adv_loss_sum = sum(log.get('adv_loss', 0) for log in logging_outputs)
    ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    metrics.log_scalar('adv_loss', adv_loss_sum / ntokens / math.log(2), ntokens, round=3)


def KL_loss(p, q, pad_mask):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q = p.float(), q.float()
    dict_size = q.size(-1)
    non_pad_mask = ~pad_mask
    p = p.view(-1, dict_size)[non_pad_mask]
    q = q.view(-1, dict_size)[non_pad_mask]
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


def forward_loss(input_loss):
    """ Forward pass to get the loss. """

    model = input_loss['model']
    decoder_out = input_loss['decoder_out']
    padding_idx = input_loss['padding_idx']
    target = input_loss['target']
    pad_mask = input_loss['pad_mask']
    p = input_loss['p']

    # virtual adversarial training loss
    def get_vat_loss():
        q = model.get_normalized_probs(decoder_out, log_probs=False)
        result = KL_loss(p, q, pad_mask)
        return result

    # adversarial training loss
    def get_adv_loss():
        log_prob = model.get_normalized_probs(decoder_out, log_probs=True)
        log_prob = log_prob.view(-1, log_prob.size(-1))
        result, _ = label_smoothed_nll_loss(
            log_prob, target.view(-1, 1), model.args.label_smoothing,
            ignore_index=padding_idx, reduce=True,
        )
        return result

    if model.args.use_reg == 1:  # VAT
        loss = get_vat_loss()
    elif model.args.use_reg == 2:  # Adv
        loss = get_adv_loss()
    else:
        raise NotImplementedError('Invalid use_reg argument.')

    return loss


def update_perturbation(perturbation, lr, adv_eps):
    grad = perturbation.grad
    grad_norm = torch.norm(grad, p=2, dim=(1, 2), keepdim=True)
    grad_norm = torch.clamp(grad_norm, min=1e-6)
    grad /= grad_norm
    result = perturbation.detach().clone() + lr * grad
    result = torch.renorm(result, p=2, dim=0, maxnorm=adv_eps)
    return result


def perturb_refine(input_refine):
    model = input_refine['model']
    sample = input_refine['sample']
    adv_eps = input_refine['adv_eps']
    padding_idx = input_refine['padding_idx']
    target = input_refine['target']
    pad_mask = input_refine['pad_mask']
    p = input_refine['p']
    perturbation_enc = input_refine['perturbation_enc']
    perturbation_dec = input_refine['perturbation_dec']
    init_enc = perturbation_enc.clone()
    init_dec = perturbation_dec.clone()

    # freeze all the parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # pre-processing, set the initial perturbation
    assert model.encoder.perturbation_enc is None and model.decoder.perturbation_dec is None
    if 0 in model.args.perturbation_target:
        model.encoder.perturbation_enc = nn.Parameter(perturbation_enc, requires_grad=True)
    if 1 in model.args.perturbation_target:
        model.decoder.perturbation_dec = nn.Parameter(perturbation_dec, requires_grad=True)

    num_refine = model.args.inner_steps
    lr = model.args.inner_lr
    result_enc, result_dec = None, None
    for _ in range(num_refine):
        decoder_out = model(**sample['net_input'], adv_step=1)
        input_loss = {
            'model': model,
            'decoder_out': decoder_out,
            'padding_idx': padding_idx,
            'target': target,
            'pad_mask': pad_mask,
            'p': p,
        }
        loss = forward_loss(input_loss)
        loss.backward()

        # update perturbations
        if 0 in model.args.perturbation_target:
            temp = update_perturbation(model.encoder.perturbation_enc, lr, adv_eps)
            model.encoder.perturbation_enc = nn.Parameter(temp)
        if 1 in model.args.perturbation_target:
            temp = update_perturbation(model.decoder.perturbation_dec, lr, adv_eps)
            model.decoder.perturbation_dec = nn.Parameter(temp)

    if 0 in model.args.perturbation_target:
        result_enc = model.encoder.perturbation_enc.detach().clone()
    if 1 in model.args.perturbation_target:
        result_dec = model.decoder.perturbation_dec.detach().clone()

    for param in model.parameters():
        param.requires_grad_(True)

    # postprocessing, set the perturbation to None
    model.encoder.perturbation_enc = None
    model.decoder.perturbation_dec = None

    return result_enc, result_dec, result_enc - init_enc, result_dec - init_dec


def compute_cache_loss(model, sample, match_output, adv_samples=None, epoch=-1, padding_idx=-1):
    target = model.get_targets(sample, match_output).view(-1)
    pad_mask = target.eq(padding_idx)
    p = model.get_normalized_probs(match_output, log_probs=False).detach()  # do not require grad
    adv_eps = model.args.eps
    src_tokens = sample['net_input']['src_tokens']
    prev_output_tokens = sample['net_input']['prev_output_tokens']

    embed_weights = model.encoder.embed_tokens.weight
    embed_size = embed_weights.size(-1)
    dtype = embed_weights.dtype
    device = embed_weights.device

    def generate_perturbation():
        # random initialization
        enc, dec = None, None
        if 0 in model.args.perturbation_target:
            temp = src_tokens.size()
            enc = torch.rand([temp[0], temp[1], embed_size], dtype=dtype, device=device)
            enc = F.normalize(enc, p=2, dim=(1, 2), eps=1e-6) * adv_eps
        if 1 in model.args.perturbation_target:
            temp = prev_output_tokens.size()
            dec = torch.rand([temp[0], temp[1], embed_size], dtype=dtype, device=device)
            dec = F.normalize(dec, p=2, dim=(1, 2), eps=1e-6) * adv_eps
        return enc, dec

    alpha = model.args.ema_alpha
    sample_ids = sample['id'].tolist()
    flag = epoch > 1  # in the first epoch, compute adversarial samples for all the samples
    if flag and epoch % model.args.cache_every_epoch > 0:
        diff_enc, diff_dec = sample['adv']['src_adv'], sample['adv']['tgt_adv']

        # compute adversarial examples
        refine_enc = torch.renorm(diff_enc, p=2, dim=0, maxnorm=adv_eps)
        refine_dec = torch.renorm(diff_dec, p=2, dim=0, maxnorm=adv_eps)
    else:
        perturbation_enc, perturbation_dec = generate_perturbation()

        # adversarial gradients are already computed (i.e., epoch > 1),
        # and we need to update them, i.e., epoch % cache_every_epoch == 0
        prev_enc, prev_dec, use_ema_mask = None, None, None
        if flag:
            prev_enc, prev_dec = sample['adv']['src_adv'], sample['adv']['tgt_adv']
            use_ema_mask = sample['adv']['ema_mask'].unsqueeze(-1).unsqueeze(-1)

        input_refine = {
            'model': model,
            'sample': sample,
            'adv_eps': adv_eps,
            'padding_idx': padding_idx,
            'target': target,
            'pad_mask': pad_mask,
            'p': p,
            'perturbation_enc': perturbation_enc,
            'perturbation_dec': perturbation_dec,
        }
        refine_enc, refine_dec, diff_enc, diff_dec = perturb_refine(input_refine)

        # exponential moving average to integrate past
        if flag:
            # when use_ema_mask[i] == 1, use EMA;
            # when use_ema_mask[i] == 0, prev_enc[i] == 0, do not use EMA
            diff_enc = diff_enc * (1.0 - alpha) + prev_enc * alpha + diff_enc * alpha * (1 - use_ema_mask)
            diff_dec = diff_dec * (1.0 - alpha) + prev_dec * alpha + diff_dec * alpha * (1 - use_ema_mask)
            # recompute adversarial examples
            refine_enc += diff_enc
            refine_dec += diff_dec

        # cache the adversarial gradients
        for i, sample_id in enumerate(sample_ids):
            if adv_samples.is_valid_idx(sample_id):
                # strip the padding
                idx = src_tokens[i].ne(padding_idx).nonzero(as_tuple=False).squeeze()
                adv_samples.set_src_sample(sample_id, torch.index_select(diff_enc[i], 0, idx))
                idx = prev_output_tokens[i].ne(padding_idx).nonzero(as_tuple=False).squeeze()
                adv_samples.set_tgt_sample(sample_id, torch.index_select(diff_dec[i], 0, idx))

    # second forward pass, get the final loss
    decoder_out = model(**sample['net_input'], perturbation_enc=refine_enc,
                        perturbation_dec=refine_dec, adv_step=1)
    input_loss = {
        'model': model,
        'decoder_out': decoder_out,
        'padding_idx': padding_idx,
        'target': target,
        'pad_mask': pad_mask,
        'p': p,
    }
    loss = forward_loss(input_loss)
    return loss
