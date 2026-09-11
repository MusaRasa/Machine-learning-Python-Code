"""
    All information about the CrossEntropyLoss
    
    # Definition: is a loss funtion used in classification tasks to measure the difference between the predicated distribution and the true labels
    Key point:
       * Commonly used for multi-classification.
       * Combines Softmax and log loss in one function
       * Lower Lower loss indicates better predictions
    *** Mathmatics formula : L = -sigma(y_i*log(p_i))
    ** in pytorch nn.CrossEntropyLoss() automativally applies softmax to logits
"""

import torch as tr
import torch.nn.functional as f
def CrossEnroLoss(output,label):
    # Convert to probability
    probs = f.softmax(output,dim=1)
    # take the logarithm
    log_prob = tr.log(probs)
    # Calculate the loss with mathmatics formula
    loss = -log_prob[0][label]
    print("Loss Function ==> ",loss)
outputs =  tr.tensor([[2.3,2.2,1.2]])
label = tr.tensor([0])
CrossEnroLoss(outputs,label)