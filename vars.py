# datasettype='MNIST'
datasettype='MNIST'
if datasettype=='MNIST':
    n_channels=1
elif datasettype=='CIFAR10':
    n_channels=3
cnnlayertype='adder'
# cnnlayertype='conv'
# cnnlayertype='adder'

# n_channels=1#MNIST
# n_channels=3#CIFAR10
epochsum =100
emb_dim=16  #dimension of embedding space for latent variables (D)
emb_num=8   #number of embeddings(K)
commit_beta=0.25
# commit_beta=0.25
# NNdatatype='float16'
# NNdatatype='float16'

test_batch_size=128
train_batch_size=128