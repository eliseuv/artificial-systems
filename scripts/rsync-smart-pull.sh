#!/bin/sh

# =================CONFIGURATION =================
# Defaults
CHECK_INTERVAL=30       # Default wait time (seconds)
REMOTE_PORT=""          # Default is empty (implies system default/22)
LOCK_DIR="/tmp/rsync-smart-pull.lock"
SSH_KEY=""              # Optional: path to private key. 

# ================================================

# Initialize variables
SOURCE_ARG=""
LOCAL_DIR=""
REMOVE_MODE=false

# 1. Flexible Argument Parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--remove)
            REMOVE_MODE=true
            shift
            ;;
        -p|--port)
            REMOTE_PORT="$2"
            shift 2
            ;;
        -i|--interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [-r] [-p port] [-i interval] [user@host:source] [dest]"
            exit 1
            ;;
        *)
            # Assign positional arguments (Source first, then Dest)
            if [ -z "$SOURCE_ARG" ]; then
                SOURCE_ARG="$1"
            elif [ -z "$LOCAL_DIR" ]; then
                LOCAL_DIR="$1"
            else
                echo "Error: Too many arguments provided."
                exit 1
            fi
            shift
            ;;
    esac
done

# Validation
if [ -z "$SOURCE_ARG" ] || [ -z "$LOCAL_DIR" ]; then
    echo "Usage: $0 [-r] [-p port] [-i interval] [user@host:source_dir] [local_dest_dir]"
    exit 1
fi

# Parse "user@host:path" logic
if [[ "$SOURCE_ARG" != *":"* ]]; then
    echo "Error: Source argument must contain a colon (host:path)"
    exit 1
fi

HOST_PART="${SOURCE_ARG%%:*}"
REMOTE_DIR="${SOURCE_ARG#*:}"

if [[ "$HOST_PART" == *"@"* ]]; then
    REMOTE_USER="${HOST_PART%%@*}"
    REMOTE_HOST="${HOST_PART#*@}"
    SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"
else
    REMOTE_HOST="$HOST_PART"
    SSH_TARGET="$REMOTE_HOST"
fi

# ================================================

# Helper to build the SSH options string dynamically
# This ensures we don't pass a -p flag if no port was specified
function get_ssh_opts {
    local OPTS=""
    if [ -n "$REMOTE_PORT" ]; then
        OPTS="-p $REMOTE_PORT"
    fi
    if [ -n "$SSH_KEY" ]; then
        OPTS="$OPTS -i $SSH_KEY"
    fi
    echo "$OPTS"
}

# Function to manage the lock
function get_lock {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        trap "rm -rf '$LOCK_DIR'" EXIT
        return 0
    else
        echo "Script is already running (Lock exists at $LOCK_DIR)."
        return 1
    fi
}

function is_file_busy {
    local filepath="$1"
    local SSH_OPTS=$(get_ssh_opts)

    # Note: $SSH_OPTS is unquoted here to allow multiple flags to expand
    if ssh $SSH_OPTS "$SSH_TARGET" "lsof -t \"$filepath\" > /dev/null 2>&1"; then
        return 0 # Busy
    else
        return 1 # Free
    fi
}

# Main Execution
if ! get_lock; then
    exit 1
fi

SSH_OPTS_DISPLAY=$(get_ssh_opts)
[ -z "$SSH_OPTS_DISPLAY" ] && SSH_OPTS_DISPLAY="(Default)"

echo "---------------------------------------------"
echo "Source:      $SSH_TARGET:$REMOTE_DIR"
echo "Destination: $LOCAL_DIR"
echo "SSH Opts:    $SSH_OPTS_DISPLAY"
echo "Interval:    ${CHECK_INTERVAL}s"
if [ "$REMOVE_MODE" = true ]; then
    echo "Mode:        MOVE (Source files will be DELETED)"
else
    echo "Mode:        COPY (Source files kept)"
fi
echo "---------------------------------------------"

while true; do
    SSH_OPTS=$(get_ssh_opts)

    # Get list: Oldest files first
    FILES_CMD="find $REMOTE_DIR -maxdepth 1 -type f -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-"
    
    # Run SSH command with dynamic options
    FILES=$(ssh $SSH_OPTS "$SSH_TARGET" "$FILES_CMD" 2>/dev/null)

    SAVEIFS=$IFS
    IFS=$'\n'
    FILE_LIST=($FILES)
    IFS=$SAVEIFS

    for REMOTE_FILE in "${FILE_LIST[@]}"; do
        [ -z "$REMOTE_FILE" ] && continue

        FILENAME=$(basename "$REMOTE_FILE")
        LOCAL_FILE="$LOCAL_DIR/$FILENAME"

        # Check if file exists locally
        if [ -f "$LOCAL_FILE" ]; then
            continue 
        fi

        echo "Checking: $FILENAME..."

        if is_file_busy "$REMOTE_FILE"; then
            echo "  > [BUSY] File is open. Skipping."
        else
            echo "  > [FREE] File ready. Syncing..."
            
            # Build Rsync Options
            RSYNC_OPTS="-Pazvhm --timeout=60"
            if [ "$REMOVE_MODE" = true ]; then
                RSYNC_OPTS="$RSYNC_OPTS --remove-source-files"
            fi

            # Construct the -e string. We use the same opts logic.
            # We trim leading spaces just in case to be clean.
            SSH_STRING="ssh $SSH_OPTS"
            
            rsync $RSYNC_OPTS -e "$SSH_STRING" \
                "$SSH_TARGET:$REMOTE_FILE" \
                "$LOCAL_DIR"

            if [ $? -eq 0 ]; then
                if [ "$REMOVE_MODE" = true ]; then
                    echo "  > [OK] Transfer complete & Source deleted."
                else
                    echo "  > [OK] Transfer complete."
                fi
            else
                echo "  > [ERR] Transfer failed."
            fi
        fi
    done

    sleep "$CHECK_INTERVAL"
done
