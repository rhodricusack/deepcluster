
# Parallelisation
For training - already dealt with by DataParallel command. Distributes mini-batches across GPUS. May scale to multiple GPUs using PyTorch NCCL

# Cluster
## Need tools to set up and autoscale cluster

TODO: Investigate suitability of [aws-parallel-cluster](https://github.com/aws/aws-parallelcluster)

### Configuration of single machines

#### AMI
AMI is defined in spot*.json file. Currently uses DLAMI with everything installed, then deletes unwanted stuff in user_data_script.sh

TODO: Should start from DLAMI Base (e.g., eu-west-1, v19.3) ami-0b5a7638f94ff56a8

#### user_data_script.sh
Configures machine on startup. Clones repository from github, attaches data drive, installs components, starts job


