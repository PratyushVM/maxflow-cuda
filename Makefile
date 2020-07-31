CC = g++
NVCC = nvcc

all : maxflow

maxflow : obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o
	$(NVCC) obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o -o maxflow

obj/main.o : src/main.cu obj
	$(NVCC) -c src/main.cu -o obj/main.o

obj/graph_s.o : src/graph_s.cpp obj
	$(CC) -c src/graph_s.cpp -o obj/graph_s.o

obj/io_par.o : src/io_par.cu obj
	$(NVCC) -c src/io_par.cu -o obj/io_par.o

obj/preflow.o : src/preflow.cu obj
	$(NVCC) -c src/preflow.cu -o obj/preflow.o 

obj/push_relabel.o : src/push_relabel.cu obj
	$(NVCC) -c src/push_relabel.cu -o obj/push_relabel.o 

obj/push_relabel_kernel.o : src/push_relabel_kernel.cu obj
	$(NVCC) -c src/push_relabel_kernel.cu -o obj/push_relabel_kernel.o 

obj/global_relabel.o : src/global_relabel.cu obj
	$(NVCC) -c src/global_relabel.cu -o obj/global_relabel.o

obj :
	mkdir obj

clean :
	rm obj/*.o ./maxflow
	rmdir obj