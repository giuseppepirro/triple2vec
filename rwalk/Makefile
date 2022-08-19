CC=/usr/local/opt/llvm/bin/clang # Set compiler if CC is not set

CFLAGS= -I/usr/local/opt/llvm/include -fopenmp -fPIC -O3 -D NDEBUG -Wall -Werror

#LDFLAGS = -L/usr/local/opt/llvm/lib

all: librwalk.so

ERRORE vanno fatti in sequenza: li ho fatti dalla linea di  comando come segue:

1) /usr/local/opt/llvm/bin/clang -c -I/usr/local/opt/llvm/include -fopenmp -fPIC -O3 -D NDEBUG -Wall -v -Werror rwalk.c -o rwalk.o
2) /usr/local/opt/llvm/bin/clang -I/usr/local/opt/llvm/include -fopenmp -fPIC -O3 -D NDEBUG -Wall -v -Werror -shared -Wl, rwalk.o -o librwalk.cpython-37m-darwin.so

L'estensione librwalk.cpython-37m-darwin.so deve contenere cpython-37m-darwin per poter funzionare in python

librwalk.so: rwalk.o
	#$(CC) $(CFLAGS) -shared -Wl, librwalk.so -o librwalk.so rwalk.o
	$(CC) $(CFLAGS) -shared -Wl, rwalk.o -o librwalk.so
	rm rwalk.o

rwalk.o: rwalk.c
	$(CC) -c $(CFLAGS) rwalk.c -o rwalk.o

clean :
	rm -rf librwalk.so rwalk.o __pycache__


[ 0  2  4  5  6  8 12 16 18 21 22]
[6 5 4 8 9 8 1 0 2 5 1 6 5 8 7 6 6 7 0 5 4 3]

[0  2  5  6  7  9 13 17 19 22 24]
[6 5 4 8 9 8 9 1 0 2 5 1 6 5 8 7 6 6 7 1 0 5 4 3]
