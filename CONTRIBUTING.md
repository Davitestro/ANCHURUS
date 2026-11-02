# Contributing to ANCHURUS

Thank you for your interest in contributing to ANCHURUS!

ANCHURUS is an experimental lightweight monocular depth estimation project focused on making depth prediction more accessible on hardware with limited computational resources.

Contributions of all kinds are welcome, including bug reports, performance improvements, architecture experiments, documentation, testing, and new ideas.

## 1. Getting Started

Before contributing, make sure you have:

- Python 3.9 or newer.
- Git.
- PyTorch.
- The dependencies listed in `requirements.txt`, if available.
- Basic familiarity with computer vision and deep learning.

### Clone the repository

```bash
git clone https://github.com/Davitestro/ANCHURUS.git
cd ANCHURUS
```

Install the project dependencies according to the repository instructions.

## 2. Types of Contributions

We welcome contributions in the following areas:

### Model Architecture
- Lightweight encoder and decoder architectures.
- Improvements to inference efficiency.
- Memory consumption optimization.
- Alternative feature extraction methods.
- Knowledge distillation improvements.

### Training
- Training pipeline improvements.
- Loss function experiments.
- Dataset preprocessing.
- Training stability and reproducibility.
- Evaluation methodology.

### Performance Optimization
- CPU inference optimization.
- GPU acceleration.
- ONNX or other deployment formats.
- Reduced memory footprint.
- Real-time inference improvements.

### Documentation
- Usage examples.
- Installation instructions.
- Architecture explanations.
- Performance reports.
- Tutorials and visual demonstrations.

### Bug Reports
Report reproducible issues involving:
- Incorrect depth predictions.
- Runtime errors.
- Hardware compatibility.
- Installation problems.
- Unexpected memory or performance behavior.

## 3. Development Guidelines

Please follow these principles when modifying the project:

- Keep implementations simple and maintainable.
- Avoid unnecessary dependencies.
- Preserve existing functionality unless the change explicitly requires otherwise.
- Document significant architectural or training changes.
- Prefer reproducible experiments over undocumented performance claims.
- Keep experimental implementations separate from stable inference code when appropriate.

## 4. Machine Learning Experiment Guidelines

Contributions involving model training or evaluation should include sufficient information to reproduce the experiment.

Whenever possible, provide:

- Dataset name and version.
- Training and validation split methodology.
- Model architecture.
- Input resolution.
- Loss functions.
- Optimizer and learning rate.
- Batch size and number of epochs.
- Hardware configuration.
- Evaluation metrics.

### Evaluation

Performance improvements should be supported by measurable results.

Depending on the experiment, useful metrics include:

- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
- AbsRel (Absolute Relative Error).
- Delta accuracy.
- Inference latency.
- FPS.
- Model size.
- Peak memory consumption.

When comparing models, use the same evaluation dataset, preprocessing, input resolution, and measurement conditions.

Do not report results from training data as validation or test performance.

### Performance Benchmarks

For inference-related contributions, specify:

- CPU or GPU model.
- Operating system.
- Input resolution.
- Precision (FP32, FP16, etc.).
- Batch size.
- Warm-up procedure.
- Measurement methodology.

Distinguish between latency and throughput. FPS alone may not accurately represent single-image inference performance.

## 5. Code Style

Please maintain the existing project structure and coding conventions.

General recommendations:

- Use descriptive variable and function names.
- Avoid unnecessary code duplication.
- Keep functions focused on a single responsibility.
- Handle invalid inputs and runtime errors appropriately.
- Avoid hardcoded machine-specific paths.
- Do not commit generated datasets, checkpoints, or large temporary files unless explicitly required.

## 6. Submitting Changes

### Step 1: Fork the repository

Create your own fork of ANCHURUS.

### Step 2: Create a branch

Use a descriptive branch name:

```bash
git checkout -b feature/description
```

Examples:

```text
feature/cpu-optimization
feature/new-loss-function
fix/inference-error
docs/installation-guide
```

### Step 3: Implement your changes

Make focused changes and test them before submitting.

### Step 4: Commit your changes

Use clear commit messages:

```bash
git add .
git commit -m "Improve CPU inference performance"
```

### Step 5: Open a Pull Request

Submit your changes against the main repository.

Include:

- A clear description of the contribution.
- The motivation behind the change.
- Relevant implementation details.
- Testing performed.
- Benchmark results, if applicable.
- Screenshots or visual comparisons when useful.

## 7. Pull Request Guidelines

Before submitting a Pull Request:

- Verify that the project runs with your changes.
- Check that existing functionality is not unintentionally broken.
- Include relevant documentation updates.
- Avoid unrelated changes in the same Pull Request.
- Disclose known limitations.

For model-related changes, include experimental evidence whenever possible.

Contributions that introduce architectural changes or new training methodologies should explain their reasoning and expected benefits.

## 8. Reporting Issues

When opening an issue, provide as much relevant information as possible.

Include:

- A description of the problem.
- Steps to reproduce it.
- Expected behavior.
- Actual behavior.
- Python and PyTorch versions.
- Hardware configuration.
- Relevant error messages or logs.

For prediction-related issues, example input images and corresponding outputs are especially helpful.

## 9. Feature Proposals

Before implementing major changes, consider opening an issue to discuss the proposal.

This is particularly useful for:

- New model architectures.
- Changes to the training pipeline.
- New dependencies.
- Major inference optimizations.
- Changes to model input or output formats.

This helps keep contributions aligned with the project's goals.

## 10. License

By contributing to ANCHURUS, you agree that your contributions will be distributed under the same license as the project.

Please ensure that you have the right to submit any code, datasets, images, or other materials included in your contribution.

## 11. Acknowledgments

Every meaningful contribution helps improve ANCHURUS.

Whether you are fixing a small bug, improving inference performance, experimenting with a new architecture, or improving documentation, your work is appreciated.

Thank you for helping make lightweight monocular depth estimation more accessible!

---

**Project:** [Davitestro/ANCHURUS](https://github.com/Davitestro/ANCHURUS)
