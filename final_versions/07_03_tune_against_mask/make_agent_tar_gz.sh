archive_name="${1:-"submission.tar.gz"}"
tar --exclude="outputs" --exclude="submission.tar.gz" --exclude="__pycache__" --exclude="cpp" --exclude="log" --exclude="models" \
    --use-compress-program="gzip -9" -cvf "$archive_name" *
echo "Agent saved to $archive_name"