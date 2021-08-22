#include <Python.h>
const int max_length = 440;
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <time.h>

using namespace std;

// Get maximal score and trace it
double  max_score (double up, double diag, double left, char * ptr){
    double  max = 0;

    if( diag >= up && diag >= left ){
        max = diag;
        *ptr = '\\';
    }
    else if( up > left){
        max = up;
        *ptr = '|';
    }
    else{
        max = left;
        *ptr = '-';
    }
    return  max;
}

// Get maximal score and trace it
double  max_score_x (double diag, double up, char * ptr){
    double  max = 0;

    if( diag >= up){
        max = diag;
        *ptr = '\\';
    }
    else{
        max = up;
        *ptr = '|';
    }
    return  max;
}

// Get maximal score and trace it
double  max_score_y (double diag, double left, char * ptr){
    double  max = 0;

    if(diag >= left ){
        max = diag;
        *ptr = '\\';
    }
    else{
        max = left;
        *ptr = '-';
    }
    return  max;
}
// Initialise scoring matrix with first row and column
void  init (double** M, double** Mx, double** My, char** M_tb, char** Mx_tb, char** My_tb, int A_n, int B_n, double gap, double gap_ext){
    M[0][0] =  0;
    Mx[0][0] = -10000.0;
    My[0][0] = -10000.0;
    M_tb[0][0] = 'n';
    Mx_tb[0][0] = 'n';
    My_tb[0][0] = 'n';

    int i=0, j=0;

    for( j = 1; j <= A_n; j++ ){
        M[0][j] = - ( gap + ( gap_ext * j ) ); // manually apply affine gap
        Mx[0][j] = -10000.0;
        My[0][j] = -10000.0;
        M_tb[0][j] =  '-';
        Mx_tb[0][j] =  '-';
        My_tb[0][j] =  '-';
    }
    for( i = 1; i <= B_n; i++ ){
        M[i][0] = - ( gap + ( gap_ext * i ) ); // manually apply affine gap
        Mx[i][0] = -10000.0;
        My[i][0] = -10000.0;
        M_tb[i][0] =  '|';
        Mx_tb[i][0] =  '|';
        My_tb[i][0] =  '|';
    }
}

// Needleman and Wunsch algorithm
int alignment (double* match_score,  double* margin_score_FP, double* margin_score_FN, double** M, double** Mx, double** My, char** M_tb, char** Mx_tb, char** My_tb, string A, string B, string& A_al, string& B_al, int A_n, int B_n, double gap, double gap_ext){
    char ptr;
    char ptrx;
    char ptry;
    int i = 0, j = 0;

    int calc_point = 0; 
    double stmp;
    for( i = 1; i <= B_n; i++ ){
        for( j = 1; j <= A_n; j++ ){

            Mx[i][j] = max_score_x(M[i - 1][j] - gap - gap_ext + margin_score_FN[calc_point], Mx[i - 1][j] - gap_ext + margin_score_FN[calc_point], &ptrx);
            Mx_tb[i][j] = ptrx;
            My[i][j] = max_score_y(M[i][j - 1] - gap - gap_ext + margin_score_FN[calc_point], My[i][j - 1] - gap_ext + margin_score_FN[calc_point], &ptry);
            My_tb[i][j] = ptry;
            stmp = M[i-1][j-1] + match_score[calc_point] + margin_score_FP[calc_point];
            M[i][j] = max_score(Mx[i][j], stmp, My[i][j], &ptr); // get max score for current optimal global alignment
            M_tb[i][j] = ptr;
            calc_point += 1;
        }
    }

    int level = 0;//0->M, 1->Mx, 2->My 
    i--; j--;
    while( i > 0 || j > 0 ){
        if(level == 0){
            switch( M_tb[i][j] )
            {
                case '|' :      level = 1;
                                break;

                case '-' :      level = 2;
                                break;

                case '\\':      A_al += A[j-1];
                                B_al += B[i-1];
                                i--;  j--;
            }
        }
        else if(level == 1){
            if(Mx_tb[i][j] == '\\'){level = 0;}
            A_al += '-';
            B_al += B[i-1];
            i--;
        }
        else if(level == 2){
            if(My_tb[i][j] == '\\'){level = 0;}
            A_al += A[j-1];
            B_al += '-';
            j--;
        }
    }

    reverse( A_al.begin(), A_al.end() );
    reverse( B_al.begin(), B_al.end() );

    delete[] match_score;
    delete[] margin_score_FN;
    delete[] margin_score_FP;
    return  0 ;
}

// Print the scoring matrix
void  print_mtx (double** M, string A, string B, int A_n, int B_n){
    cout << "        ";
    for( int j = 0; j < A_n; j++ ){
        cout << A[j] << "   ";
    }
    cout << "\n  ";

    for( int i = 0; i <= B_n; i++ ){
        if( i > 0 ){
            cout << B[i-1] << " ";
        }
        for( int j = 0; j <= A_n; j++ ){
            cout.width( 3 );
            cout << M[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Print the traceback matrix
void  print_tb (char** M_tb, string A, string B, int A_n, int B_n){
    cout << "        ";
    for( int j = 0; j < A_n; j++ ){
        cout << A[j] << "   ";
    }
    cout << "\n  ";

    for( int i = 0; i <= B_n; i++ ){
        if( i > 0 ){
            cout << B[i-1] << " ";
        }
        for( int j = 0; j <= A_n; j++ ){
            cout.width( 3 );
            cout << M_tb[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Initiate matrices, align and export
int** NW (double* match_score, double* margin_score_FP, double* margin_score_FN, string A, string B, string& A_al, string& B_al, int A_n, int B_n, double gap, double gap_ext, bool print_align, bool print_mat){
    int align_nuc = 150;
    // Create alignment matrix
    double** M = new double* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        M[i] = new double [A_n+1];
    }
    // Create alignment matrix
    double** Mx = new double* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        Mx[i] = new double [A_n+1];
    }
    // Create alignment matrix
    double** My = new double* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        My[i] = new double [A_n+1];
    }
    
    // Create traceback matrix
    char** M_tb = new char* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        M_tb[i] = new char [A_n+1];
    }
    // Create traceback matrix
    char** Mx_tb = new char* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        Mx_tb[i] = new char [A_n+1];
    }
    // Create traceback matrix
    char** My_tb = new char* [B_n+1];
    for( int i = 0; i <= B_n; i++ ){
        My_tb[i] = new char [A_n+1];
    }



    clock_t t; // for timing execution
    t = clock(); // get time of start

    // Initialize traceback and F matrix (fill in first row and column)
    init (M, Mx, My, M_tb, Mx_tb, My_tb, A_n, B_n, gap, gap_ext);


    // Create alignment
    alignment (match_score, margin_score_FP, margin_score_FN, M, Mx, My, M_tb, Mx_tb, My_tb, A, B, A_al, B_al, A_n, B_n, gap, gap_ext);

    t = clock() - t; // get time when finished
    // double score = M[B_n][A_n]; // get alignment score

    if(print_mat == 1){
        print_mtx(M, A, B, A_n, B_n);
        print_tb(M_tb, A, B, A_n, B_n);
    }

    if(print_align == 1){
        cout << endl << "Alignments:" << endl;
        int start = 0; // start of new line for printing alignments
        int cntr = 0; // iterator for printing alignments
        int Al_n = A_al.length(); // length of alignment
        do{
            cout << start+1 << " A: ";
            for (cntr = start; cntr < start+align_nuc; cntr++){
                if(cntr < Al_n){
                    cout << A_al[cntr];
                }else{
                    break;
                }
            }
            cout << " " << cntr << endl << start+1 << " B: ";
            for (cntr = start; cntr < start+align_nuc; cntr++){
                if(cntr < Al_n){
                    cout << B_al[cntr];
                }else{
                    break;
                }
            }
            cout << " " << cntr << endl << endl;
            start += align_nuc;
        }while(start <= Al_n);
    }

    // Free memory
    for( int i = 0; i <= B_n; i++ )  delete M[i];
    delete[] M;
    for( int i = 0; i <= B_n; i++ )  delete Mx[i];
    delete[] Mx;
    for( int i = 0; i <= B_n; i++ )  delete My[i];
    delete[] My;
    for( int i = 0; i <= B_n; i++ )  delete M_tb[i];
    delete[] M_tb;
    for( int i = 0; i <= B_n; i++ )  delete Mx_tb[i];
    delete[] Mx_tb;
    for( int i = 0; i <= B_n; i++ )  delete My_tb[i];
    delete[] My_tb;
    return 0;
}

static PyObject* c_multiply_list(PyObject* c_list, PyObject* d_list, PyObject* e_list, int n, string A, string B, int lengthA, int lengthB, double gap_open, double gap_extend, int print_align, int print_mat)
{
    double *match_score;
    match_score = new double[n];
    double *margin_score_FP;
    margin_score_FP = new double[n];
    double *margin_score_FN;
    margin_score_FN = new double[n];


    PyObject* item, *common_index_A_B;
    string A_al, B_al = "";

    for (int i = 0; i < n; i++){
        item = PyList_GetItem(c_list, i);
        match_score[i] = PyFloat_AsDouble(item);    // PyObject -> float 
        // Py_DECREF(item);
    }
    Py_DECREF(c_list);

    for (int i = 0; i < n; i++){
        item = PyList_GetItem(d_list, i);
        margin_score_FP[i] = PyFloat_AsDouble(item);    // PyObject -> float 
        // Py_DECREF(item);
    }
    Py_DECREF(d_list);

    for (int i = 0; i < n; i++){
        item = PyList_GetItem(e_list, i);
        margin_score_FN[i] = PyFloat_AsDouble(item);    // PyObject -> float 
        // Py_DECREF(item);
    }
    Py_DECREF(e_list);

    NW(match_score, margin_score_FP, margin_score_FN, A, B, A_al, B_al, lengthA, lengthB, gap_open, gap_extend, print_align, print_mat);
    // delete[] match_score;

    common_index_A_B = PyList_New(max_length * 2);
    int A_index = 0;
    int B_index = 0;
    for (int i = 0; i < A_al.length(); i++){
        if (A_al[i] == '-'){
            item = Py_BuildValue("l", 0);  // long -> PyObject
            PyList_SetItem(common_index_A_B, B_index + max_length, item);
            B_index += 1;
        }
        else if (B_al[i] == '-'){
            item = Py_BuildValue("l", 0);  // long -> PyObject
            PyList_SetItem(common_index_A_B, A_index, item);
            A_index += 1;
        }
        else{
            item = Py_BuildValue("l", 1);  // long -> PyObject
            PyList_SetItem(common_index_A_B, B_index + max_length, item);
            item = Py_BuildValue("l", 1);  // long -> PyObject
            PyList_SetItem(common_index_A_B, A_index, item);
            B_index += 1;
            A_index += 1;
        }
    }
    for (; A_index < max_length; A_index++){
        item = Py_BuildValue("l", 0);  // long -> PyObject
        PyList_SetItem(common_index_A_B, A_index, item);
        }
    for (; B_index < max_length ; B_index++){
        item = Py_BuildValue("l", 0);  // long -> PyObject
        PyList_SetItem(common_index_A_B, B_index + max_length, item);
        }

    return common_index_A_B;
}

// C function "get list and return list"
static PyObject* py_list(PyObject* self, PyObject* args)
{
    int n;
    char* seq1, *seq2;
    int print_align, print_mat;
    int lengthA, lengthB;
    double gap_open, gap_extend;
    PyObject* c_list;
    PyObject* d_list;
    PyObject* e_list;

    // decide type (list)
    if (!PyArg_ParseTuple(args, "OOOssiiddii", &c_list, &d_list, &e_list, &seq1, &seq2, &lengthA, &lengthB, &gap_open, &gap_extend, &print_align, &print_mat)){
        return NULL;
    }
    string A = seq1;
    string B = seq2;

    // Check list
    if PyList_Check(c_list){
        // get length of the list
        n = PyList_Size(c_list);
    }else{
        return NULL;
    }

    // Check list
    if PyList_Check(d_list){
        // get length of the list
        n = PyList_Size(d_list);
    }else{
        return NULL;
    }

    // Check list
    if PyList_Check(e_list){
        // get length of the list
        n = PyList_Size(e_list);
    }else{
        return NULL;
    }

    return c_multiply_list(c_list, d_list, e_list, n, A, B, lengthA, lengthB, gap_open, gap_extend, print_align, print_mat);
}

// Function Definition struct
static PyMethodDef MultiplyList[] = {
    { "global_aln", py_list, METH_VARARGS, "global alignment "},
    { NULL }
};

// Module Definition struct
static struct PyModuleDef alignment_C = {
    PyModuleDef_HEAD_INIT,
    "alignment_C",
    "Python3 C API Module for alignment",
    -1,
    MultiplyList
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_alignment_C(void)
{
    return PyModule_Create(&alignment_C);
}