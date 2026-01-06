## IC9600 benchmark (ICC regression variant)

This repo includes an IC9600 benchmark implementation under `icc/ic9600_benchmark/`.
It is intended to validate the ICC “complexity” signal on a standard image-complexity benchmark using a **ConvNeXt-V2 regression** model and ICNet-style metrics (RMSE/RMAE/Pearson/Spearman).

### Get IC9600

IC9600 is a cornerstone benchmark for **automatic image complexity assessment**, and it is the basis of our standard-benchmark validation for ICC. Please follow the authors’ official repository for dataset access and setup instructions:
- `https://github.com/tinglyfeng/IC9600`

You need a local directory with:
- `train.txt` (lines: `image_name␠␠score` using the IC9600 **double-space** separator)
- `test.txt`  (same format)
- `images/` directory containing referenced images

Example layout:
```
IC9600/
  train.txt
  test.txt
  images/
    000001.jpg
    000002.jpg
    ...
```

### Train

```bash
python -m icc.ic9600_benchmark.train \
  --ic9600_train_txt /path/to/IC9600/train.txt \
  --ic9600_test_txt  /path/to/IC9600/test.txt \
  --ic9600_img_dir   /path/to/IC9600/images \
  --output_dir runs/ic9600_convnext_regression \
  --model_name convnextv2_nano.fcmae_ft_in22k_in1k \
  --img_size 512 \
  --batch_size 32 \
  --num_epochs 30 \
  --learning_rate 1e-4
```

Outputs:
- `runs/.../checkpoints/best.ckpt` (best on `val_pearson`)
- `runs/.../checkpoints/last.ckpt`
- `runs/.../checkpoints/final_model.ckpt`
- `runs/.../run_config.json`

### Evaluate

```bash
python -m icc.ic9600_benchmark.evaluate \
  --checkpoint runs/ic9600_convnext_regression/checkpoints/best.ckpt \
  --ic9600_test_txt /path/to/IC9600/test.txt \
  --ic9600_img_dir  /path/to/IC9600/images \
  --output_dir runs/ic9600_convnext_regression/eval_best
```

Outputs:
- `metrics.json` (RMSE, RMAE, Pearson, Spearman)
- `predictions.csv` (filename, ground_truth, prediction, error, abs_error)

### Reference

```bibtex
@article{feng2023ic9600,
  title={IC9600: A Benchmark Dataset for Automatic Image Complexity Assessment},
  author={Feng, Tinglei and Zhai, Yingjie and Yang, Jufeng and Liang, Jie and Fan, Deng-Ping and Zhang, Jing and Shao, Ling and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  number={01},
  pages={1--17},
  year={2023},
  publisher={IEEE Computer Society},
  doi={10.1109/TPAMI.2022.3232328},
}
```
