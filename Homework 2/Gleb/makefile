CC = g++
CFLAGS = -g -std=c++17 -lstdc++fs $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null)
DEPS = *.hpp *.h

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: task1 task2 task3

task1: task1.o hog_visualization.o
	$(CC) -o $@ $^ $(CFLAGS)

task2: task2.o hog_visualization.o RandomForest.o features.o
	$(CC) -o $@ $^ $(CFLAGS)

task3: task3.o hog_visualization.o RandomForest.o features.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	find . -maxdepth 1 -type f -name "*.o" -delete
	find . -maxdepth 1 -type f -not -name "*.*" -not -name "makefile" -delete
	
