## Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

This is a fork of the software project that accompanies the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**, 
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

<p align="center">
  <img src="data/depth-pro-goose-pointcloud-teaser.gif" alt="Depth Pro point cloud demo" />
  <br />
  <sub><sup>RGB and near-infared point cloud generated from a single image using Depth Pro</sup></sub>
</p>

The authors present a foundation model for zero-shot metric monocular depth estimation. The model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.


The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

## Getting Started

The original authors recommend setting up a virtual environment. Using e.g. miniconda, the `depth_pro` package can be installed via:

```bash
conda create -n depth-pro -y python=3.9
conda activate depth-pro

pip install -e .
```

I created a `Dockerfile` description that you can use to run the inference in a container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/git/ml-depth-pro:/workspace/ml-depth-pro \
    -w /workspace/ml-depth-pro \
    depthpro
```

To download pretrained checkpoints follow the code snippet below:
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

### Running from commandline

The original authors provide a helper script to directly run the model on a single image:
```bash
# Run prediction on a single image:
depth-pro-run -i input/2023-03-02_garching__0018_1677754323813048729_windshield_vis.png -o output/
# Run `depth-pro-run -h` for available options.
```

I've added the `depth2pcd.py` script to create a PCD file to visualize the depth map in viewers like CloudCompare:
```bash
# construct a PCD file with RGB colors per point from the original input image
# and the Depth Pro depth map prediction stored as .npz file
python depth2pcd.py ./output/2023-03-02_garching__0018_1677754323813048729_windshield_vis.npz  \
  --focal 1780 \
  --labels ./input/2023-03-02_garching__0018_1677754323813048729_labelids.png \
  --exclude-labels 8,53 \
  --erode-semantic-boundary 0 \
  --erode-depth-boundary 3 \
  --depth-threshold 0.025 \
  --image ./input/2023-03-02_garching__0018_1677754323813048729_windshield_vis.png \
  --output ./2023-03-02_garching__0018_1677754323813048729_windshield_vis.pcd

# construct a PCD file with near-infrared (NIR) point values as 'intensity' point features
python depth2pcd.py ./output/2023-03-02_garching__0018_1677754323813048729_windshield_vis.npz \
  --focal 1780 \
  --labels ./input/2023-03-02_garching__0018_1677754323813048729_labelids.png \
  --exclude-labels 8,53 \
  --erode-semantic-boundary 0 \
  --erode-depth-boundary 3 \
  --depth-threshold 0.025 \
  --nir ./input/2023-03-02_garching__0018_1677754323813048729_windshield_nir.png \
  --output ./2023-03-02_garching__0018_1677754323813048729_windshield_nir.pcd
```

## Citation

If you find Depth Pro useful, please cite the following paper:

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

## License
This sample code is released under the [LICENSE](LICENSE) terms.

The model weights are released under the [LICENSE](LICENSE) terms.

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
