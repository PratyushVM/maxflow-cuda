CC = g++
NVCC = nvcc

# run make all to create ./maxflow executable
all : maxflow

# run make debug to create ./maxflowdbg executable, which is the debug build to use in cuda-gdb and cuda-memcheck
debug : maxflowdbg

maxflow : obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o
	$(NVCC) obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o -o maxflow

maxflowdbg : obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o
	$(NVCC) obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o -g -G -o maxflowdbg

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

# run make clean to clean obj files and the executable(s)
clean :

	if [ obj ] ; \
	then \
		rm obj/*.o ; \
		rmdir obj ; \
	fi ; \
	if [ maxflow ] ; \
	then \
		rm ./maxflow ; \
	fi ;
	if [ maxflowdbg ] ; \
	then \
		rm ./maxflowdbg ; \
	fi ;