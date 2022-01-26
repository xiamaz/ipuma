all: 
	g++ -std=c++11 single_compilation_serialize.cpp -lpoplar -o serialize
	g++ -std=c++11 single_compilation_deserialize.cpp -lpoplar -o deserialize
	touch ./resources/exec.json
	touch ./resources/graph.json
	touch ./resources/tensor.json
	touch ./resources/exec.poplar_exec
	touch ./resources/graph.cbor
	touch ./resources/tensor.cbor

clean:
	rm -f serialize
	rm -f deserialize
	rm -f ./resources/exec.poplar_exec
	rm -f ./resources/graph.json
	rm -f ./resources/tensor.json
	rm -f ./resources/exec.json
	rm -f ./resources/graph.cbor
	rm -f ./resources/tensor.cbor