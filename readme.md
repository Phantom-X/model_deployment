nohup uvicorn app:app --host 127.0.0.1 --port 8008 > server.log 2>&1 &

ps aux | grep uvicorn

kill -9 PID