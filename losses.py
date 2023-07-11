import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss, mse_loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_loss = bce_loss(logits_real, torch.ones_like(logits_real).to(device))
    fake_loss = bce_loss(logits_fake, torch.zeros_like(logits_fake).to(device))
    loss = real_loss + fake_loss
    ##########       END      ##########
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
      
    ##########       END      ##########
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake).to(device))
    return loss

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_loss = 0.5 * torch.mean((scores_real - 1)**2)
    fake_loss = 0.5 * torch.mean((scores_fake)**2)
    loss =  real_loss + fake_loss
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
      
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = 0.5 * torch.mean((scores_fake - 1)**2)
    ##########       END      ##########
    
    return loss


## Extra Credit 
def wass_discriminator_loss(scores_real, scores_fake):
    loss = -torch.mean(scores_real) + torch.mean(scores_fake)
    return loss


def wass_generator_loss(scores_fake):
    loss = -torch.mean(scores_fake)
    return loss
