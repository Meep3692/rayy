CFLAGS = -I.\SDL2-2.0.14\include -L.\SDL2-2.0.14\lib\x64 -lSDL2 -DSDL_MAIN_HANDLED -O3

rayy.exe: main.cu
	nvcc -o rayy main.cu $(CFLAGS)
	
clean:
	rm -f rayy.exe rayy.lib rayy.exp

all: rayy.exe