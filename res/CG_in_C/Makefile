CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -g -c
LDFLAGS = -lm
ifeq ($(OS), Windows_NT)
	TARGET = solver.exe
else
	TARGET = solver.out
endif

all: $(TARGET)

$(TARGET): main.o
	$(CC) $< $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm *.o && rm $(TARGET)

