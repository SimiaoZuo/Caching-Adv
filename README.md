# ARCH: Adversarial Regularization with Caching

Code for [ARCH: Efficient Adversarial Regularized Training with Caching](https://arxiv.org/abs/2109.07048), Findings of EMNLP 2021.

## Run the code
  
### Dependency
* The most convenient way to run the code is to use this docker image: `tartarusz/adv-train:azure-pytorch-apex-v1.7.0`. 
  The image supports running on Microsoft Azure.
* Our implementation is modified from the [Fairseq](https://github.com/pytorch/fairseq) code base.
  
### Instructions
* Please refer to the [Fairseq examples](https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md)
for dataset pre-processing.
* Run `pip install -e .` to install locally.
* Use `bash get_nearest_samples.sh [path-to-checkpoint]` to pre-compute a neighbor file.
Here, `path-to-checkpoint` is any pre-trained model.
* Use `bash run.sh` to run the code. To use random neighbors instead of pre-computed ones, 
  remove the `--neighbor-file` argument and add a `--prop-neighbors [prop]` argument to randomly select `prop` indices.

### Note
* The major modification from the original Fairseq code base is the following.
  * `fairseq/criterions/cache_loss.py` is the main file that handles caching.
  * `adv_dataset.py` stores and constructs the adversarial perturbations.
  * `fairseq/models/transformer.py` modifies embedding to include adversarial perturbations.
  * `fairseq/tasks/fairseq_task.py` contains the adversarial training procedure.
  

## Reference

Please cite the following paper if you use this code.

```
@article{zuo2021arch,
  title={ARCH: Efficient Adversarial Regularized Training with Caching},
  author={Zuo, Simiao and Liang, Chen and Jiang, Haoming and He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2109.07048},
  year={2021}
}
```