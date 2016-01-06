#include <iostream>
#include <stdlib.h> 
#include <time.h> 

using namespace std;

//Returns the sum of an array
int arraysum(int somearray[], int size) {
	//TODO: PARALLELIZE
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += somearray[i];
	}
	return sum;
}

//Returns the max of an array
int arraymax(int somearray[], int size) {
	//TODO: PARALLELIZE
	int max = 0;
	for (int i = 0; i < size; i++) {
		if (somearray[i] > max) max = somearray[i];
	}
	return max;
}

int main(int argc, char *argv[]) {
	
	//Get a random seed
	srand(time(NULL));

	//Create array and populate it
	int size = atoi(argv[1]);
	int myarray[size];
	for (int i = 0; i < size; i++) {
		myarray[i] = 1 + rand() % 1000;
	}
	
	//Run functions in parallel
	//TODO: PARALLELIZE
	int mysum = arraysum(myarray, size);
	int mymax = arraymax(myarray, size);

	//Output the answers
	cout << "Maximum: " << mymax << " Sum: " << mysum << endl;
	return 0;
}