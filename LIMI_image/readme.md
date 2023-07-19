# LIMI for image data
The scripts are stored in the folder **expsh**. 

Getting Started as follows:

1. `pip install _req_img.txt`

2. train the tested models (the models are stored in `./exp/classifier`)

    `bash train_model.sh`

4. generate $\mathbf{Z}_{init}$ latent vector and corresponding instances (stored in `./exp/img/`)

    `bash generate.sh`

5. use the tested models to predict instances in 3. (stored in `./exp/predict_attrs` )

   `bash predict.sh`

6. train the surrogate boundary to imitate the real decision boundary (stored in `./exp/train_boundaries/`)

   `bash train_boundary.sh`

7. conduct the probing process to generate potential individual discriminatory images. (stored in `./exp/main_fair/`)

   `bash main_fair.sh`

7. conduct fairness testing to check whether the potential image is a discriminatory instance. (the discriminatory instances will be stored in `./exp/model_test/`)

    `bash model_test.sh`