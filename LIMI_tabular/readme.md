# LIMI for tabular data
The scripts are stored in the folder **expsh**. 

Getting Started as follows:

1. `pip install _requirements.txt`

    make sure `copulas==0.7.0`, `ctgan==0.5.1`, `rdt==0.6.3`

2. train the tested models (the models are stored in `./exp/train_dnn/train/`)

    `bash train_dnn.sh`

3. train the gan model (the gans are stored in `./exp/gans/`)
    `bash train_gan.sh`

4. generate $\mathbf{Z}_{init}$ latent vector and corresponding instances (stored in `./exp/table/`)

    `bash generate.sh`

5. use the tested models to predict instances in 4. (stored in the folder of each model by default )

   `bash predict.sh`

6. train the surrogate boundary to imitate the real decision boundary (stored in `./exp/train_boundaries/`)

   `bash train_boundary.sh`

7. conduct the fairness testing to generate natural individual discriminatory instances. (stored in `./exp/main_fair/ours[_description]`)

   `bash main_fair.sh`

   And the instances are stored in  `global_samples.npy`