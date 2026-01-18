#!/bin/bash

# Get the mode from environment (ui or processor)
MODE=${MODE:-ui}

echo "CITB4 Application"
echo "========================"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "processor" ]; then
    # Job mode - run the processor
    echo "Starting Cloud Run Job Processor..."
    exec python job_processor.py
else
    # Service mode - run Flask UI
    PORT=${PORT:-5000}
    echo "Starting web server on port $PORT..."
    exec python app.py
fi
