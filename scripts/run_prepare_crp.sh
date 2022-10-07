python3 -m experiments.prepare_crp --model_name vgg16 --dataset_name imagenet --batch_size 48
python3 -m experiments.prepare_crp --model_name vgg16_corrupted --dataset_name imagenet_corrupted --batch_size 48
python3 -m experiments.prepare_crp --model_name vgg16_uncorrupted --dataset_name imagenet_corrupted --batch_size 48