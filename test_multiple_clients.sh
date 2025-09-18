#!/bin/bash

echo "Starting Data Stream Server..."
python3 stream_server.py --interval 1 &
SERVER_PID=$!

sleep 2

echo "Starting Client 1..."
python3 stream_client.py --id Client-1 &
CLIENT1_PID=$!

sleep 1

echo "Starting Client 2..."
python3 stream_client.py --id Client-2 &
CLIENT2_PID=$!

sleep 1

echo "Starting Client 3..."
python3 stream_client.py --id Client-3 &
CLIENT3_PID=$!

echo ""
echo "Server PID: $SERVER_PID"
echo "Client 1 PID: $CLIENT1_PID"
echo "Client 2 PID: $CLIENT2_PID"
echo "Client 3 PID: $CLIENT3_PID"
echo ""
echo "Press Ctrl+C to stop all processes"

trap "kill $SERVER_PID $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID 2>/dev/null; exit" INT

wait