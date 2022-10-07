sample_ids=(
4419 #test 1
16887 #test 2
)
ds=imagenet
model=vgg16

for id in "${sample_ids[@]}"
do
  common_arguments="--model_name $model --dataset_name $ds --sample $id"
  python3 -m experiments.crp_ $common_arguments --layer_name features.28 --mode relevance
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode relevance
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode activation
  python3 -m experiments.cam $common_arguments --layer_name features.29
  python3 -m experiments.lrp $common_arguments
  python3 -m experiments.integrated_grad $common_arguments
  python3 -m experiments.shap $common_arguments
done


sample_ids=(
# corrupted
28759 34304 49086 48841 35973 13155 49086 24302 21477 44856 7504 29870 4827 35901 3763 12365 19944 17704 33279 43870 14970 28351 34304 15405 28759 1898 19311 35271 37306 46404 12282 27327 25931
)
ds=imagenet_corrupted
model=vgg16_corrupted

for id in "${sample_ids[@]}"
do
  common_arguments="--model_name $model --dataset_name $ds --sample $id"
  python3 -m experiments.crp_ $common_arguments --layer_name features.28 --mode relevance
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode relevance
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode activation
  python3 -m experiments.cam $common_arguments --layer_name features.29
  python3 -m experiments.lrp $common_arguments
  python3 -m experiments.integrated_grad $common_arguments
  python3 -m experiments.shap $common_arguments
done

sample_ids=(
# not corrupted
48805 35950 13198 49086 24313 21454 44896 7522 29890 4837 35918 3792 12365 19938 17704 33275 43871 14994 28360 34339 15420 28780 1876 19306 35267 37322 46402 12287 27347 25929
)

ds=imagenet_corrupted
model=vgg16_uncorrupted

for id in "${sample_ids[@]}"
do
  common_arguments="--model_name $model --dataset_name $ds --sample $id"
  python3 -m experiments.crp_ $common_arguments --layer_name features.28 --mode relevance --class_specific_samples False
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode relevance
  python3 -m experiments.concept_based $common_arguments --layer_name features.28 --mode activation
  python3 -m experiments.cam $common_arguments --layer_name features.29
  python3 -m experiments.lrp $common_arguments
  python3 -m experiments.integrated_grad $common_arguments
  python3 -m experiments.shap $common_arguments
done