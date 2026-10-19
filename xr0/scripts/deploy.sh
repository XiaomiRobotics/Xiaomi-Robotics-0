export TOKENIZERS_PARALLELISM=false

conda activate robot

# Check if exactly 3 arguments are provided (model path, number of ports, and number of GPUs)
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_path> <number_of_ports> <number_of_gpus>"
    echo "Example: $0 /path/to/model 3 2  # Will start 3 servers on ports 10086-10088 across 2 GPUs"
    exit 1
fi

# First argument is the model path
MODEL_PATH=$1

# Second argument is the number of ports
NUM_PORTS=$2

# Third argument is the number of GPUs
NUM_GPUS=$3

# Validate that NUM_PORTS is a positive integer
if ! [[ "$NUM_PORTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of ports must be a positive integer"
    exit 1
fi

# Validate that NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of GPUs must be a positive integer"
    exit 1
fi

# Starting port
BASE_PORT=10086

# Session name
SESSION_NAME="model_servers"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

# Function to set up environment in a pane
setup_environment() {
    local target=$1
    tmux send-keys -t $target "export TOKENIZERS_PARALLELISM=false" Enter
    tmux send-keys -t $target "conda activate robot" Enter
}

# Start servers, creating new windows as needed
for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$((i % NUM_GPUS))  # Distribute servers across GPUs
    
    if [ $i -eq 0 ]; then
        # Use the first window (already created)
        WINDOW_NAME="server-$(printf "%02d" $i)"
        tmux rename-window -t $SESSION_NAME:0 $WINDOW_NAME
        setup_environment "$SESSION_NAME:$WINDOW_NAME"
        tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID ZE_AFFINITY_MASK=$GPU_ID python3 mibot/server/deploy.py --model '$MODEL_PATH' --port $PORT" Enter
    else
        # Create new windows for additional servers
        WINDOW_NAME="server-$(printf "%02d" $i)"
        tmux new-window -d -t $SESSION_NAME -n $WINDOW_NAME
        setup_environment "$SESSION_NAME:$WINDOW_NAME"
        tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID ZE_AFFINITY_MASK=$GPU_ID python3 mibot/server/deploy.py --model '$MODEL_PATH' --port $PORT" Enter
    fi
done

# Attach to the session
echo "Started $NUM_PORTS servers in tmux session '$SESSION_NAME'"
echo "Ports: $(seq -s ', ' $BASE_PORT $((BASE_PORT + NUM_PORTS - 1)))"
echo "Distributed across $NUM_GPUS GPUs"
echo "Use 'tmux attach -t $SESSION_NAME' to view the servers"
