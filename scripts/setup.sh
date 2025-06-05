set -e  # Exit on any error

echo "Setting up CLIP-Fields Thor Integration..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"


setup_thor_env() {
    echo "Setting up AI2-THOR environment..."

    # Create conda environment
    conda create -n thor python=3.9 -y

    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate thor

    # Install PyTorch with CUDA support
    # conda install pandas pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

    # Install AI2-THOR
    pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246

    conda install pandas scipy matplotlib -n thor
    pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

    # Install other requirements
    pip install -r "$PROJECT_ROOT/thor_env/requirements.txt" --no deps

    echo "AI2-THOR environment setup complete!"
}

# Function to create and setup CLIP-Fields environment
setup_clipfields_env() {
    echo "Setting up CLIP-Fields environment..."

    # Create conda environment
    conda create -n clipfields python=3.9 -y

    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate clipfields

    conda install pandas pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.8 -c pytorch-lts -c nvidia -y

    # Install requirements
    pip install -r "$PROJECT_ROOT/clipfields_env/requirements.txt" --no deps

    # Clone and install CLIP-Fields dependencies
    echo "Installing CLIP-Fields dependencies..."

    # Create temporary directory for cloning
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Clone CLIP-Fields repository
    git clone --recursive https://github.com/notmahi/clip-fields.git
    cd clip-fields

    # Install gridencoder
    cd gridencoder
    # Set CUDA_HOME if not set
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        echo "Set CUDA_HOME to: $CUDA_HOME"
    fi
    python setup.py install
    cd ..

    # Copy necessary files to our project
    cp -r Detic "$PROJECT_ROOT/clipfields_env/"
    cp -r LSeg "$PROJECT_ROOT/clipfields_env/"
    cp grid_hash_model.py "$PROJECT_ROOT/clipfields_env/"
    cp misc.py "$PROJECT_ROOT/clipfields_env/"

    # Clean up
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"

    echo "CLIP-Fields environment setup complete!"
}

# Function to download model weights
download_models() {
    echo "Downloading model weights..."

    # Create models directory
    mkdir -p "$PROJECT_ROOT/models"
    cd "$PROJECT_ROOT/models"

    # Download Detic weights (if not already present)
    if [ ! -f "detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k.pth" ]; then
        echo "Downloading Detic weights..."
        wget https://dl.fbaipublicfiles.com/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k.pth
    fi

    # Create checkpoints directory for LSeg
    mkdir -p "$PROJECT_ROOT/checkpoints"
    cd "$PROJECT_ROOT/checkpoints"

    # Download LSeg weights (if not already present)
    if [ ! -f "demo_e200.ckpt" ]; then
        echo "Downloading LSeg weights..."
        # Note: Replace with actual LSeg checkpoint URL
        echo "Please manually download LSeg checkpoint from the official repository"
        echo "and place it at: $PROJECT_ROOT/checkpoints/demo_e200.ckpt"
    fi

    cd "$PROJECT_ROOT"
}

# Function to run tests
run_tests() {
    echo "Running tests..."

    # Test AI2-THOR environment
    echo "Testing AI2-THOR environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate thor
    python -c "import ai2thor; print('AI2-THOR import successful')"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

    # Test CLIP-Fields environment
    echo "Testing CLIP-Fields environment..."
    conda activate clipfields
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import clip; print('CLIP import successful')"
    python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers import successful')"

    echo "All tests passed!"
}

# Main setup process
main() {
    echo "Starting setup process..."

    # Parse command line arguments
    SETUP_THOR=true
    SETUP_CLIPFIELDS=true
    DOWNLOAD_MODELS=true
    RUN_TESTS=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-thor)
                SETUP_THOR=false
                shift
                ;;
            --skip-clipfields)
                SETUP_CLIPFIELDS=false
                shift
                ;;
            --skip-models)
                DOWNLOAD_MODELS=false
                shift
                ;;
            --skip-tests)
                RUN_TESTS=false
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-thor        Skip AI2-THOR environment setup"
                echo "  --skip-clipfields  Skip CLIP-Fields environment setup"
                echo "  --skip-models      Skip model weight downloads"
                echo "  --skip-tests       Skip test execution"
                echo "  -h, --help         Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Setup environments
    if [ "$SETUP_THOR" = true ]; then
        setup_thor_env
    fi

    if [ "$SETUP_CLIPFIELDS" = true ]; then
        setup_clipfields_env
    fi

    # Download models
    if [ "$DOWNLOAD_MODELS" = true ]; then
        download_models
    fi

    # Run tests
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi

    echo ""
    echo "Setup complete! 🎉"
    echo ""
    echo "To use the integration:"
    echo "1. Start the CLIP-Fields server:"
    echo "   conda activate clipfields"
    echo "   cd $PROJECT_ROOT/clipfields_env"
    echo "   python server.py"
    echo ""
    echo "2. In another terminal, run AI2-THOR navigation:"
    echo "   conda activate thor"
    echo "   cd $PROJECT_ROOT/thor_env"
    echo "   python thor_integration.py"
    echo ""
}

# Run main function
main "$@"

