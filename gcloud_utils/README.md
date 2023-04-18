# Running in GCloud
https://embracingtherandom.com/deep-learning/cloud/tensorflow/docker/dockerise-your-tf/#do-ya-got-some-gpu-quota

```
gcloud compute instances create t4-instance-1 \
    --project=northern-bot-383915 \
    --zone=us-west1-b \
    --machine-type=n1-standard-4 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=767806385608-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=t4-instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230411-debian-10-py37,mode=rw,size=50,type=projects/northern-bot-383915/zones/us-west1-b/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=ec-src=vm_add-gcloud \
    --reservation-affinity=any \
	--metadata-from-file startup-script=./startup.sh
```