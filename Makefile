CC = mpicc
CFLAGS = -O3 -ffast-math -Wall
LIBS = -lmpi -lm
OBJS = qr_fac.o gram_schmidt.o functions.o

qr_fac: qr_fac.c $(OBJS)
	$(CC) $(CFLAGS) -o qr_fac $(OBJS) $(LIBS)
	
qr_fac.o: qr_fac.c functions.h
	$(CC) $(CFLAGS) -c qr_fac.c $(LIBS)
	
gram_schmidt: gram_schmidt.c gram_schmidt.h
	$(CC) $(CFLAGS) -c gram_schmidt.c $(LIBS)
	
functions.o: functions.c functions.h
	$(CC) $(CFLAGS) -c functions.c -lm

clean:
	rm -f qr_fac $(OBJS)