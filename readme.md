不使用Eureka服务发现框架 启动命令 ： nohup uvicorn app:app --host 127.0.0.1 --port 8008 > server.log 2>&1 &

使用Eureka服务发现框架 启动命令 ： nohup python app.py > server.log 2>&1 &

ps aux | grep uvicorn

kill -9 PID