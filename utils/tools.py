import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


from scipy import cluster as clst
from sklearn.decomposition import PCA
from IPython.display import clear_output

def cluster_based(representations, n_cluster: int, n_pc: int):
  """ Improving Isotropy of input representations using cluster-based method
      Args: 
            inputs:
                  representations: 
                    input representations numpy array(n_samples, n_dimension)
                  n_cluster: 
                    the number of clusters
                  n_pc: 
                    the number of directions to be discarded
            output:
                  isotropic representations (n_samples, n_dimension)

            """


  #representations = representations.cpu().detach().numpy()
  centroid, label=clst.vq.kmeans2(representations, n_cluster, minit='points',
                                  missing='warn', check_finite=True)
  cluster_mean=[]
  for i in range(max(label)+1):
    sum=np.zeros([1,2048]);
    for j in np.nonzero(label == i)[0]:
      #print("representations[j].shape",representations[j].shape)
      #print("sum.shape",sum.shape)
      sum=np.add(sum, representations[j])
    cluster_mean.append(sum/len(label[label == i]))

  zero_mean_representation=[]
  for i in range(len(representations)):
    zero_mean_representation.append((representations[i])-cluster_mean[label[i]])

  cluster_representations={}
  for i in range(n_cluster):
    cluster_representations.update({i:{}})
    for j in range(len(representations)):
      if (label[j]==i):
        cluster_representations[i].update({j:zero_mean_representation[j]})

  cluster_representations2=[]
  for j in range(n_cluster):
    cluster_representations2.append([])
    for key, value in cluster_representations[j].items():
      cluster_representations2[j].append(value)

  cluster_representations2=np.array(cluster_representations2)


  model=PCA()
  post_rep=np.zeros((representations.shape[0],representations.shape[1]))

  for i in range(n_cluster):
      model.fit(np.array(cluster_representations2[i]).reshape((-1,2048)))
      component = np.reshape(model.components_, (-1, 2048))

      for index in cluster_representations[i]:
        sum_vec = np.zeros((1, 2048))

        for j in range(n_pc):
                sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                          np.transpose(component)[:,j].reshape((2048,1))) * component[j]
        
        post_rep[index]=cluster_representations[i][index] - sum_vec

  clear_output()

  return post_rep




def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    #f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    #f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    #f_a_norm = f_a
    #f_b_norm = f_b

    #element_wise = 0.5 * (0 - torch.log(f_a.std()) + f_a.std() / 1 + (f_a.mean() - 0).pow(2) / 1 - 1)
    #kl_1 = element_wise.sum(-1)

    #element_wise_ = 0.5 * (0 - torch.log(f_b.std()) + f_b.std() / 1 + (f_b.mean() - 0).pow(2) / 1 - 1)
    #kl_2 = element_wise_.sum(-1)
    ###return kl

    #element_wise = 0.5 * (0 - torch.log(f_a.std(1)) + f_a.std(1) / 1 + (f_a.mean(1) - 0).pow(2) / 1 - 1)
    #kl_1_ = element_wise.sum(-1)

    #element_wise_ = 0.5 * (0 - torch.log(f_b.std(1)) + f_b.std(1) / 1 + (f_b.mean(1) - 0).pow(2) / 1 - 1)
    #kl_2_ = element_wise_.sum(-1)

    f_a_norm = torch.Tensor(cluster_based(f_a.cpu().detach().numpy(),1,1))
    f_b_norm = torch.Tensor(cluster_based(f_b.cpu().detach().numpy(),1,1))

    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss
