# Adaptive Cellular Automata Edge Detection

## Overview
This project implements edge detection using adaptive cellular automata (CA) that dynamically adjust their behavior based on local image characteristics. Unlike traditional edge detection methods that use fixed parameters, this approach uses a 3-state CA system that adapts to local contrast and intensity variations.

## Project Structure
```
adaptive_ca_edge_detection/
├── src/
│   ├── main.py          # Entry point and orchestration
│   ├── ca.py            # Core cellular automata logic
│   └── utils.py         # Helper functions
├── tests/
│   └── test_ca.py       # Unit tests
├── data/
│   ├── input_images/    # Input grayscale images
│   └── output_images/   # Generated edge maps
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks for experimentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup
1. Clone the repository
2. Create virtual environment: `python3 -m venv .venv`
3. Activate environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
```bash
python src/main.py --input data/input_images/sample.png --output data/output_images/edges.png
```

## Development Status
- [x] Project structure created
- [x] Dependencies installed
- [ ] Core CA logic implementation
- [ ] Adaptive thresholding system
- [ ] Testing framework
- [ ] Performance optimization
- [ ] Documentation and examples

## Technical Approach
The system uses a 3-state cellular automata where each cell can be:
- **State 0**: Non-edge (background)
- **State 1**: Potential edge (candidate)
- **State 2**: Confirmed edge

The CA adapts its transition rules based on:
- Local intensity gradients
- Gaussian-weighted neighborhood statistics
- Dynamic thresholding parameters

## Next Steps
1. Implement core CA logic in `src/ca.py`
2. Create adaptive thresholding utilities
3. Build testing framework
4. Add performance benchmarks against traditional methods