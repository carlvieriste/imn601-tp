.SUFFIXES:
.SUFFIXES: .o .cpp

CCC        = g++
CCFLAGS    = -g -fpermissive
LIBS       = -lm
EXECUTABLE = all
OBJETS     = MImage.o gc/GCoptimization.o gc/LinkedBlockList.o gc/graph.o gc/maxflow.o


all:tp1A tp1B tp1C tp1D tp1E

tp1A: tp1A.o $(OBJETS)
	$(CCC) -g -o $@ $@.o $(OBJETS) $(LIBS)

tp1B: tp1B.o $(OBJETS)
	$(CCC) -g -o $@ $@.o $(OBJETS) $(LIBS)

tp1C: tp1C.o $(OBJETS)
	$(CCC) -g -o $@ $@.o $(OBJETS) $(LIBS)

tp1D: tp1D.o $(OBJETS)
	$(CCC) -g -o $@ $@.o $(OBJETS) $(LIBS)

tp1E: tp1E.o $(OBJETS)
	$(CCC) -g -o $@ $@.o $(OBJETS) $(LIBS)
	
.cpp.o:
	$(CCC) $(CCFLAGS) -c $< -o $@ 

clean:
	rm tp1A tp1B tp1C tp1D tp1E *~ *.o *.exe
