# **SSF: Sparse Long-Range Scene Flow for Autonomous Driving**  

[![ArXiv](https://img.shields.io/badge/arXiv-2501.17821-blue.svg)](https://arxiv.org/abs/2501.17821)  

This repository contains the official implementation of **Sparse Scene Flow (SSF)**, a scalable and efficient approach to **long-range 3D scene flow estimation**. By leveraging **sparse convolutions**, SSF overcomes the computational challenges of existing methods, enabling **efficient scene flow estimation at distances beyond 50 meters**.  

🚀 **What’s New in SSF?**  
✅ **Sparse Feature Fusion:** A novel approach to handle unordered sparse voxel features between consecutive point clouds.  
✅ **SSF Metric:** A new **range-wise evaluation metric** that measures sensitivity of scene flow performance with respect to distance from ego-vehicle, prioritizing long-range points and addressing gaps in existing benchmarks.  

---

## **📦 Requirements**  

SSF builds upon [SeFlow](https://github.com/KTH-RPL/SeFlow). Follow its installation guide **[here](https://github.com/KTH-RPL/SeFlow/tree/6f20f8f87c114f80a355307a2508f71d42765383)**.

**Additional Dependencies:**  
- [`spconv 2.3.6`](https://github.com/traveller59/spconv)  
- [`torch_scatter 2.1.2`](https://github.com/rusty1s/pytorch_scatter)  

---

## **🚀 Training SSF**  

### **Step 1: Data Preparation**  
Prepare the dataset following [SeFlow’s instructions](https://github.com/KTH-RPL/SeFlow/tree/6f20f8f87c114f80a355307a2508f71d42765383?tab=readme-ov-file#data-preparation).  

### **Step 2: Train the Model**  
Train SSF using:  
```bash
python 1_train.py model=ssf loss_fn=deflowLoss train_data=<PATH_TO_TRAIN_DATA> val_data=<PATH_TO_VAL_DATA>
```
💡 Tip: Set `train_data` and `val_data` in `conf/config.yaml` for faster iteration. See `assets/slurm/1_train.sh` for distributed training recipe.

---

## **📝 Inference**  
Run inference using:
```bash
python 2_eval.py dataset_path=<PATH_TO_DATASET> checkpoint=<PATH_TO_CKPT> av2_mode=(val,test) save_res=(True,False)
```
💡 Tip: Modify `conf/eval.yaml` for easier experimentation.

---

## **📊 SSF Metric (New!)**
We introduce a new distance-based evaluation metric for scene flow estimation. Below is an example output for SSF, `point_cloud_range=204.8`m, `voxel_size=0.2`m

| Distance  | Static    | Dynamic  | NumPointsStatic | NumPointsDynamic |
|-----------|----------|----------|-----------------|------------------|
| 0-35      | 0.00836  | 0.11546  | 3.33e+08        | 1.57e+07         |
| 35-50     | 0.00910  | 0.16805  | 4.40e+07        | 703125           |
| 50-75     | 0.01107  | 0.20448  | 3.25e+07        | 395398           |
| 75-100    | 0.01472  | 0.24133  | 1.31e+07        | 145281           |
| 100-inf   | 0.01970  | 0.30536  | 1.32e+07        | 171865           |
| **Mean**  | 0.01259  | 0.20693  | NaN             | NaN              |

---

## **🎨 Visualization**  
Generate scene flow for visualization: 
```bash
python 3_vis.py checkpoint=<PATH_TO_CKPT> dataset_path=<PATH_TO_DATASET>
```
💡 Tip: Modify params in `conf/vis.yaml` for parameter tuning.

### **🆕 Multi-Scene Flow Visualizer**
Visualize multiple scene flows **side by side** using `vis_multiple()`. Pass a list of flows to `--flow_mode`:
```bash
python tools/scene_flow.py --flow_mode <YOUR_FLOW_MODE> --data_dir <PATH_TO_DATASET>
```

By default the script calls `vis()` which means single scene flow, and `flow_mode` is set to `'flow'` which means ground truth flow.

---

## **💡 Citation and Acknowledgements** 
If SSF is useful for your research, please cite our paper using:

```
@article{khoche2025ssf,
  title={SSF: Sparse Long-Range Scene Flow for Autonomous Driving},
  author={Khoche, Ajinkya and Zhang, Qingwen and Sanchez, Laura Pereira and Asefaw, Aron and Mansouri, Sina Sharif and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2501.17821},
  year={2025}
}
```

Sincere thanks to:

- 👩‍🔬 [Qingwen Zhang](https://kin-zhang.github.io/) for her work on [Seflow](https://github.com/KTH-RPL/SeFlow) and her help with SSF metric and manuscript revision. 
- 🔬 The authors of [FSD](https://github.com/tusen-ai/SST.git) for their valuable contributions to the field. 

This work was supported by the research grant [PROSENSE](https://www.vinnova.se/en/p/prosense-proactive-sensing-for-autonomous-driving/) funded by [VINNOVA](https://www.vinnova.se/en). The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.