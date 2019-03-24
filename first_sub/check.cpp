#include <iostream>
#include <string>
using namespace std;

int main(int argc, char**argv) {
    //Read which event to run from arguments, default is event # 1000
    //Supports events 0-124 for test, or 1000-1099 for validation (small dataset)
    char file[1000];
    string base_path = argv[1];
    cout << argv[1] << endl;
    sprintf(file, "%s/adjacency", argv[1]);
    FILE *fp = fopen(file, "r");
    if (!fp) {
        cout << "Could not open adjacency" << endl;
        cout << file << endl;
        exit(0);
    }
    else{
        cout << "Opened" << endl;
    }
    fclose(fp);
    }
