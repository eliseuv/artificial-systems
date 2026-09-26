#!/bin/sh

# =================CONFIGURATION =================
# Remote Server Details
REMOTE_USER="evf"
REMOTE_HOST="desktop-8d6io5l.local"
REMOTE_PORT="2222"
REMOTE_DIR="Projects/artificial-systems/data/"
SSH_KEY="/home/evf/.ssh/id_ed25519"

# Local Details
LOCAL_DIR="/run/media/evf/Research/contact_process/new/dani/"

# Settings
CHECK_INTERVAL=30       # Seconds to sleep between main loop cycles
LOCK_DIR="/tmp/rsync_pull.lock"

# ================================================

function get_lock {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        trap "rm -rf '$LOCK_DIR'" EXIT
        return 0
    else
        echo "Script is already running (Lock exists)."
        return 1
    fi
}

# Function to check if file is open on remote host
# Returns 0 (true) if file is BUSY
# Returns 1 (false) if file is FREE
function is_file_busy {
    local filepath="$1"
    # Check if lsof sees the file open (quiet mode)
    if ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "lsof -t \"$filepath\" > /dev/null 2>&1"; then
        return 0 # Busy
    else
        return 1 # Free
    fi
}

if ! get_lock; then
    exit 1
fi

echo "Starting Sorted Rsync Watcher (Oldest -> Newest)..."

while true; do
    # 1. Get list of files sorted by modification time (Oldest First)
    # Explanation of the command run remotely:
    # find ... -printf '%T@ %p\n' : Print "Timestamp(seconds) /path/to/file"
    # sort -n                     : Sort numerically (Smallest/Oldest timestamp at top)
    # cut -d' ' -f2-              : Remove the timestamp column, keep the path
    
    FILES_CMD="find $REMOTE_DIR -maxdepth 1 -type f -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-"
    
    FILES=$(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "$FILES_CMD" 2>/dev/null)

    # Convert newline-separated string to array
    SAVEIFS=$IFS
    IFS=$'\n'
    FILE_LIST=($FILES)
    IFS=$SAVEIFS

    for REMOTE_FILE in "${FILE_LIST[@]}"; do
        [ -z "$REMOTE_FILE" ] && continue

        FILENAME=$(basename "$REMOTE_FILE")
        LOCAL_FILE="$LOCAL_DIR/$FILENAME"

        # Check if we already have the file locally
        if [ -f "$LOCAL_FILE" ]; then
            continue 
        fi

        echo "Checking candidate: $FILENAME..."

        # 2. Check using lsof
        if is_file_busy "$REMOTE_FILE"; then
            echo "  [BUSY] File is currently open by a process. Skipping."
            # Since we are sorting oldest first, if the oldest is busy, 
            # we continue to the next oldest.
        else
            echo "  [FREE] File is closed. Starting transfer..."
            
            rsync -Pazvhm --remove-source-files --timeout=60 -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
                "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FILE" \
                "$LOCAL_DIR"

            if [ $? -eq 0 ]; then
                echo "  Transfer complete: $FILENAME"
            else
                echo "  Transfer failed: $FILENAME"
            fi
        fi
    done

    sleep "$CHECK_INTERVAL"
done
