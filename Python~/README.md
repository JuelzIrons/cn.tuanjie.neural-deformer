# Neural Deformer Trainer Python Package

The **Neural Deformer Trainer** Python package is a machine learning toolkit designed to efficiently approximate complex mesh deformations for character animations in real-time applications. It is primarily used within the **Tuanjie Engine Editor**.

For more details on how to use this package within the Tuanjie Engine Editor, please refer to the official documentation [here](https://docs.unity.cn/cn/Packages-cn/cn.tuanjie.neural-deformer@1.0/manual/).

This package predicts vertex displacements on mesh surfaces from joint rotations, enabling high-fidelity animations while minimizing computational overhead compared to traditional geometric deformation methods.

Specifically, the inputs and outputs are as follows:

* **Inputs**: Joint rotations represented as quaternions.
* **Outputs**: Displacements of each vertex on the mesh surface.

## Installation

### 1. Set up the Environment

Ensure that the `uv` package is installed in your environment. If not, follow the installation guide [here](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Hardware-specific Dependencies

Please note that the `pyproject.toml` file is dynamically generated based on `Editor/NeuralDeformerTrainerEditor.cs`. Choose the appropriate dependencies for your hardware when installing. Below are the example dependencies:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neural_deformer_trainer"
version = "1.0.0"
description = "Neural Deformer Trainer: A neural network model that can predict accurate outfit deformations on 3D characters in motion."
readme = "README.md"
requires-python = ">=3.10,<3.13"
dependencies = [
    "numpy<2",
    "ipykernel>=6.29.5",
    "matplotlib>=3.10.1",
    "onnx>=1.17.0",
    "onnxscript>=0.2.5",
    "tensorboardx>=2.6.2.2",
    "torch==2.2.0",
    "torch-kmeans>=0.2.0"
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

Please adjust this file according to your hardware configuration.

### 3. Install the Package

To install the **Neural Deformer** package, navigate to the Python package directory of the project and run the following command:

```bash
pip install .
```

This command will install the package and all its dependencies.


## How to Train the Model

### Step 1: Run the Training Script

Execute the following command to start training the model:

```bash
uv run train.py
```

### Step 2: Optional Arguments

You can customize the training process using various arguments. To view all available options, run:

```bash
uv run train.py -h
```
