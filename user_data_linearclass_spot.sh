#!/bin/bash
# To base64 encode, need to run 
# echo `base64 user_data_linearclass_spot.sh -w0` 
# and insert into JSON


# Get instance ID, Instance AZ, Volume ID and Volume AZ 
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=eu-west-1

# Get imagenet volume from matching AZ. Wierdly, if you use the JSON format you get "AND" between filters; if you use text, OR 
JSON='[{"Name":"tag:Name","Values":["imagenet"]}, {"Name":"availability-zone","Values":["'"$INSTANCE_AZ"'"]}]'

VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filters "$JSON" --query "Volumes[].VolumeId" --output text)

# Proceed if Volume Id is not null or unset
if [ $VOLUME_ID ]; then
		# Check if the Volume AZ and the instance AZ are same or different.
		# If they are different, create a snapshot and then create a new volume in the instance's AZ.

		# Get imagenet snapshot
		SNAPSHOT_ID=$(aws ec2 describe-snapshots --region $AWS_REGION --filter 'Name=tag:Name,Values=imagenet' --query "Snapshots[].SnapshotId" --output text)

		# Attach volume to instance
		aws ec2 attach-volume \
			--region $AWS_REGION --volume-id $VOLUME_ID \
			--instance-id $INSTANCE_ID --device /dev/sdi
		sleep 10

		# Mount volume and change ownership, since this script is run as root
		mkdir /imagenet
		mount /dev/xvdi /imagenet
		chown -R ubuntu: /imagenet/

		# Get training code
		cd /home/ubuntu/
		git clone https://github.com/rhodricusack/deepcluster.git
#		chown -R ubuntu: deepcluster
#		cd deepcluster/

		# Install FAISS
#		sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate pytorch_p27; conda install faiss-gpu cudatoolkit=10.0 -c pytorch --yes"
# Not needed for training
        # Make some disc space
		cd /home/ubuntu/anaconda3/envs
		rm -rf amazonei* tensorflow* cntk* mxnet* caffe* chainer* theano* &
        

		# Initiate training using the tensorflow_36 conda environment
		sudo -H -u ubuntu bash -c "cd /home/ubuntu/deepcluster; ./eval_linear_spot.sh"
fi

# After training, clean up by cancelling spot requests and terminating itself
#SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
#aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances
