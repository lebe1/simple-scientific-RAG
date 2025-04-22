#!/bin/bash

# Set the target directory (default to current directory if not provided)
TARGET_DIR="${1:-.}"
DRY_RUN=0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--directory) TARGET_DIR="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -d, --directory DIR    Directory to scan (default: current directory)"
            echo "  --dry-run              Print files that would be deleted without deleting them"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Scanning directory: $TARGET_DIR"

# Count of files deleted
COUNT=0

# Find all json files that are exactly 2 bytes in size and check their content
find "$TARGET_DIR" -name "*.json" -type f -size 2c | while read -r file; do
    # Check if the file contains only '[]'
    if [[ "$(cat "$file")" == "[]" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "Would delete: $file"
        else
            echo "Deleting: $file"
            rm "$file"
            COUNT=$((COUNT + 1))
        fi
    fi
done