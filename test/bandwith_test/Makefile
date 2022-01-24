all: bandwidth_test

%: %.cpp
	g++ -std=c++11 $^ -lpoplar -o $@

clean:
	rm -f bandwidth_test