# Weighted Score Matching for Temporal Point Processes

Code for Is Score Matching Suitable for Estimating Point Processes? NeurIPS 2024

## Dependencies:
* Use python==3.8
* See `requirements.txt`

## Instructions:
* Put the `data` folder inside the root folder. The datasets are available [here](https://drive.google.com/drive/folders/1WrIXFaMd7cvxrJjXbYy3qWfDLYSRLbB_?usp=sharing)
* To reproduce Table 2 in our paper, run the files in `scripts` folder. For each experiment, we run three seeds 1,2 & 3. There are a total of 2(models) x 4(datasets) x 3(methods) x 3(seeds) = 72 experiments to be runned.
* Use commend like `bash sahp_hs_wsm.sh` to run an experiment. The results will be saved in `results` folder.
* We also provide the trained models and validation results for Table 2 in a google drive folder [here](https://drive.google.com/drive/folders/1NuWup6mbrmKYfxZ92fBDrX2erpQp-2rx?usp=sharing).

## Reference:
Please cite the following paper if you use this code.
```

```
