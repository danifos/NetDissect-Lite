#!/bin/bash

file=settings.py
quantiles=("0.004" "0.005" "0.006")

cp $file $file'.bkp'
sed -i '6c DATASET = "imagenet"' $file
sed -i '13c OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_"+DATASET+"_"+str(QUANTILE)' $file

diff_quantiles() {
	for quantile in ${quantiles[@]}
	do
		sed -i '7c QUANTILE = '$quantile $file
		python main.py
	done
}

sed -i '87c \ \ \ \ BATCH_SIZE = 16' $file
sed -i '5c MODEL = "inception_v3"' $file
diff_quantiles

sed -i '87c \ \ \ \ BATCH_SIZE = 32' $file
sed -i '5c MODEL = "vgg16"' $file
diff_quantiles

rm $file
mv $file'.bkp' $file
