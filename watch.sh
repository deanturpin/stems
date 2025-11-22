#!/bin/bash

# Simple file watcher for development - rebuilds and runs on changes
# Usage: ./watch.sh [optional: test file path]

TEST_FILE="${1:-Cesária Evora - Angola (Carl Craig remix).wav}"

echo "Watching for changes in src/ include/ CMakeLists.txt..."
echo "Will rebuild and test with: $TEST_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Initial build
make

# Watch for changes
fswatch -o src/ include/ CMakeLists.txt | while read; do
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "File changed, rebuilding..."
    echo "═══════════════════════════════════════════════════════"

    if make; then
        echo ""
        echo "Build successful! Testing..."
        echo "───────────────────────────────────────────────────────"
        if [ -f "$TEST_FILE" ]; then
            ./build/stems "$TEST_FILE"
        else
            echo "Test file not found: $TEST_FILE"
            echo "Run with: ./watch.sh <path-to-test-file>"
        fi
    else
        echo "Build failed!"
    fi

    echo ""
done
