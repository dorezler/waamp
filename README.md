# WAAMP - Workflows for the Analysis of AntiMicrobial Peptides

Ready to use ComfyUI modules to build workflows for data cleaning, ML model training and prediction in antimicrobial peptides domain.

## How to use it?

1. Clone ComfyUI repository: `git clone https://github.com/Comfy-Org/ComfyUI.git`.
2. Install ComfyUI dependencies: `cd ComfyUI && pip install -r requirements.txt`.
3. Clone this repository: `git clone https://github.com/dorezler/waamp.git`.
4. Install WAAMP dependencies: `cd waamp && pip install -r requirements.txt`.
5. Copy `waamp/custom_nodes/amp_nodes` to `ComfyUI/custom_nodes`.
6. Run ComfyUI: `python main.py --cpu` (use `--cpu` flag if GPU is not available).
7. Open exemplary JSON workflow with `File > Open` (or `Ctrl + o`) from `waamp/workflows` directory.

*Note*: Exemplary workflows are designed to parse CSV files exported from [DBAASP](https://dbaasp.org/) database.

## License

MIT License – see [LICENSE](LICENSE) file.

## Author

Dorota Rezler
