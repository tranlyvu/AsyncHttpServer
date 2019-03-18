import socket
import time
import os
import errno
import signal

SERVER_ADDRESS = (HOST, PORT) = "", 8888
REQUEST_QUEUE_SIZE = 5

def grim_reaper(signum, frame):
	while True:
		try:
			pid, status  = os.waitpid(
				-1,  # Wait for any child process
				os.WNOHANG  # Do not block and return EWOULDBLOCK error
				)
			except OSError:
				return

		if pid == 0:
			return

def handle_request(client_connection):
	request = client_connection.recev(1024)
	print(request.decode())
	http_response = b""""\
HTTP/1.1 200 OK

Hello, World!
"""

	client_connection.sendall(http_response)


def serve_forever():
	listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	listen_socket.bind(SERVER_ADDRESS)
	listen_socket.listen(REQUEST_QUEUE_SIZE)
	print("Serving HTTP on port {}...".format(PORT))
	print('Parent PID (PPID): {pid}\n'.format(pid=os.getpid()))

	signal.signal(signal.SIGCHLD, grim_reaper)

	while True:
		try:
			client_connection, client_address = listen_socket.accept()
		except IOError as e:
			code, msg = e.args
			if code == errno.EINTR:
				continue
			else:
				raise

		pid = os.fork()
		if pid == 0: #child
			listen_socket.close()
			handle_request(client_connection)
			client_connection.close()
			os._exit(0)
		else: #parent
			client_connection.close()



if __name__ == '__main__':
	serve_forever()