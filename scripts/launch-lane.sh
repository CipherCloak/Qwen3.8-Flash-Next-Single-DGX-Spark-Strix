# SPDX-License-Identifier: AGPL-3.0-or-later
load_launch_lane() {
    START_SCRIPT="$REPO_DIR/start.sh"
    LANE_MEMWATCH_MIN_GIB=""
    LANE_MEMWATCH_MIN_FREE_GIB=""
    local file="$REPO_DIR/logs/launch-lane"
    if grep -qx 'V030=true' "$file" 2>/dev/null; then
        START_SCRIPT="$REPO_DIR/start-v030.sh"
    elif grep -qx 'V034=true' "$file" 2>/dev/null; then
        START_SCRIPT="$REPO_DIR/start-v034.sh"
    else
        return 0
    fi
    LANE_MEMWATCH_MIN_GIB=$(grep -xE 'MEMWATCH_MIN_GIB=[0-9]+' "$file" | tail -1 | cut -d= -f2)
    LANE_MEMWATCH_MIN_FREE_GIB=$(grep -xE 'MEMWATCH_MIN_FREE_GIB=[0-9]+' "$file" | tail -1 | cut -d= -f2)
    return 0
}
