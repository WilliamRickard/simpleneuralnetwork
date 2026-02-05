#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<array>
#include<cstdlib>
#include<ctime>
#include<iomanip>
#include<string>
using namespace std;
void matrixmultiply(vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b);
void matrixadd(vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b);
void matrixsubtract(vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b);
void matrixScaler(vector< vector<double> > &c,double &a );
void printmatrix(vector< vector<double> > &c,string q, int d);
void setmatrixrandom(vector< vector<double> > &c, int r);
void setmatrixrandompos(vector< vector<double> > &c, int r);
void sigmoid(vector< vector<double> > &c);
void sigmoidprime(vector< vector<double> > &c);
void sigmoidAndPrime(vector< vector<double> > &z, vector< vector<double> > &a, vector< vector<double> > &aprime);
void matrixTranspose(vector< vector<double> > &b, vector< vector<double> > &c);
void textFileToMatrix(vector< vector<double> > &c, string q);
void MatrixtoTextFile(vector< vector<double> > &c, string q);
void setMatrix(vector< vector<double> > &c,double p);
void costfunc(double &costvalue,vector< vector<double> > &a,vector< vector<double> > &b);
void setcolumn(vector< vector<double> > &c,double p);
void setrow(vector< vector<double> > &c,double p);
void equatMatrix(vector< vector<double> > &a,vector< vector<double> > &b);
void hadamardproduct(vector< vector<double> > &c,vector< vector<double> > &a,vector< vector<double> > &b);
void columnadd(vector< vector<double> > &c,vector< vector<double> > &a);
void perError(int r,double &c ,vector< vector<double> > &a,vector< vector<double> > &b);
void StandardDevY(double &c ,vector< vector<double> > &a);
void errordiff(vector< vector<double> > &c,vector< vector<double> > &a,vector< vector<double> > &b);
void MaxCol(double &c ,vector< vector<double> > &a);
void testRun(int &a,int &b,int &c);
void batchOnlineRun(int &n,int &s,int it,int &exSize,int &descents,double &tar,int &rWone,int &rWtwo,bool &rw,double &nabla,double &mo);
void offlineRun(int &n,int &dc,int &s,double &tar,int &rWone,int &rWtwo,bool &rw);

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    bool batchOnline = false;
    bool offline     = false;
    bool test        = true;
    bool train       = true;                                                                                         
    int  r           = 48;
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // Overall Setup
    int    numberOfVariables       = 11;
    int    layers                  = 3;
    int    nodesInFirstHiddenLayer = 16;
    int    rangeWone               = 4;
    int    rangeWtwo               = 4;
    bool   randomiseWeights        = false;
    double percentageErrorTarget   = 3.9;
    double learningRate            = 0.0001;
    double momentum                = 0.75;
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // Batch setup
    int iterations            = 1;
    int exampleSize           = 13853;
    int numberOfDescents      = 1000000;
    int times                 = 1;
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // Offline Setup
    int datacol               = 10000;
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // Testing setup
    int xColumns              = numberOfVariables;
    int xRows                 = 13853;
    int wOneColumns           = nodesInFirstHiddenLayer;

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if( batchOnline == true ) {
        for(int i=0;i<times;i++){
        batchOnlineRun(numberOfVariables,nodesInFirstHiddenLayer,iterations,exampleSize,numberOfDescents,percentageErrorTarget,rangeWone,rangeWtwo,randomiseWeights,learningRate,momentum);
            numberOfDescents = numberOfDescents + numberOfDescents;
        }
    }
    
    if( offline == true ){
        offlineRun(numberOfVariables,datacol,nodesInFirstHiddenLayer,percentageErrorTarget,rangeWone,rangeWtwo,randomiseWeights);
    }
    
    if( test == true ) {
        testRun(xColumns,xRows,wOneColumns);
    }
}

void matrixmultiply( vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b ) {
    
    unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    
    if ( n == p) {
        for(unsigned long int i=0;i<m;i++){
            for(unsigned long int j=0;j<q;j++){
                c[i][j] = 0;
            }
            for(unsigned long int t=0;t<n;t++){
                const double a_it = a[i][t];
                const vector<double> &b_row = b[t];
                for(unsigned long int j=0;j<q;j++){
                    c[i][j] += a_it * b_row[j];
                }
            }
        }
    }
    else  {
        cout << " **** MATRIX INCOMPATIBLE **** " << endl;
    }
}
void matrixadd(vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b ){
    
    unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    if ( ( m == p ) && ( n == q ) ) { for(int i=0;i<m;i++){
        for(int j=0;j<q;j++){c[i][j] = a[i][j] + b[i][j];} } }
        else { cout << " **** MATRIX INCOMPATIBLE **** " << endl; }
}
void matrixsubtract(vector< vector<double> > &c, vector< vector<double> > &a , vector< vector<double> > &b ){
    
    unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    if ( ( m == p ) && ( n == q ) ) { for(int i=0;i<m;i++){
        for(int j=0;j<q;j++){c[i][j] = a[i][j] - b[i][j];} } }
    else { cout << " **** MATRIX INCOMPATIBLE **** " << endl; }
}
void matrixScaler(vector< vector<double> > &c,double &a ){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
     for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            c[i][j] =  ( a ) * ( c[i][j] );
        }
     }
}
void printmatrix(vector< vector<double> > &c, string q, int d){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    
    int e = q.length() ;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if( i == 0 && j == 0 ){
                if(c[i][j] > 0){
                    cout << q << " = " << setprecision(d) << fixed << c[i][j] << "    " ;
                }
                else {
                    cout << q << " = " << setprecision(d) << fixed << c[i][j] << "   " ;
                }
            }
            else if( j == 0 && ( i > 0 )){
                if(c[i][j]>0){
                    cout << endl ;
                    for(int k=0;k<e;k++){
                        cout << " "  ;
                    }
                    cout << "   " << setprecision(d) << fixed << c[i][j] << "    " ;
                }
                else {
                cout << endl ;
                    for(int k=0;k<e;k++){
                        cout << " "  ;
                    }
                cout << "   " << setprecision(d) << fixed << c[i][j] << "   " ;
                }
            }
            else {
                if(c[i][j] > 0){
                    cout << setprecision(d) <<  fixed << c[i][j] << "   " ;
                }
                else {
                cout << setprecision(d) <<  fixed << c[i][j] << "  " ;
                }
            }
        }
    }
    cout << endl;
    cout << endl;
}
void setmatrixrandom(vector< vector<double> > &c, int r) {
   
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            double k = ( rand() % 1000 ) / 1000.00;
            if( k < 0.50){
                c[i][j] = double((( rand() % ( ( r ) * ( 100000 ) ) ) + 1.00 ) * -1.00) / ( ( 100000.00 ) );
            }
            else {
                c[i][j] = double((( rand() % ( ( r ) * ( 100000 ) ) ) + 1.00 ) ) / ( ( 100000.00 ) );
            }
        }
    }
}
void setmatrixrandompos(vector< vector<double> > &c, int r){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            
                c[i][j] = double((( rand() % ( ( r ) * ( 100000 ) ) ) + 1.00 ) ) / ( ( 100000.00 ) );
        }
    }

}
void sigmoid(vector< vector<double> > &c){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            c[i][j] = (double) 1 / ( 1 + exp( ( -1.00 )  *  ( c[i][j] ) ) ) ;
        }
    }
}
void sigmoidprime(vector< vector<double> > &c){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            double inter = exp( ( - 1.00 )  *  ( c[i][j] ) );
            c[i][j] = inter / ( ( 1.00 + inter )  *  ( 1.00 + inter ) ) ;
        }
    }
}
void sigmoidAndPrime(vector< vector<double> > &z, vector< vector<double> > &a, vector< vector<double> > &aprime){
    unsigned long int m = z.size() ; unsigned long int n = z[0].size();
    for(unsigned long int i=0;i<m;i++){
        for(unsigned long int j=0;j<n;j++){
            const double sig = (double) 1 / ( 1 + exp( ( -1.00 )  *  ( z[i][j] ) ) );
            a[i][j] = sig;
            aprime[i][j] = sig * (1.00 - sig);
        }
    }
}
void matrixTranspose(vector< vector<double> > &b,vector< vector<double> > &c){
    unsigned long int m = b.size() ; unsigned long int n = b[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            b[i][j] = c[j][i] ;
        }
    }
}
void textFileToMatrix(vector< vector<double> > &c, string q){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    ifstream myfile;
        myfile.open(q);
        if(!myfile.is_open()){
            cout << "FAILED TO OPEN FILE: " << q << endl;
            return;
        }
            for(int i=0;i<m;i++)
                for(int j=0;j<n;j++) {
                    if(!(myfile >> c[i][j])){
                        cout << "INSUFFICIENT OR INVALID DATA IN FILE: " << q << endl;
                        return;
                    }
        }
    myfile.close();
}
void MatrixtoTextFile(vector< vector<double> > &c, string q){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    ofstream myfile;
    myfile.open(q);
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++) {
            if(j== n-1){
                myfile << setprecision(10) << c[i][j] << endl;
            }
            else{
            myfile << setprecision(10) << c[i][j] << " " ;
            }
        }
    myfile.close();
}
void setMatrix(vector< vector<double> > &c,double p){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            c[i][j] = p ;
        }
    }
}
void costfunc(double &costval,vector< vector<double> > &a,vector< vector<double> > &b) {
    unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    double k = 0;
    double rr= 0;
    if( ( m == p ) && ( n == q ) ){
        for(int i=0;i<m;i++){
            k = a[i][0] - b[i][0];
            rr = ( ( k ) * ( k ) ) + rr;
        }
        costval = (1.000/2.000) * ( rr ) ;
    }
    else {
        cout << "COST CANNOT BE COMPUTED, INPUT DIM =//= OUTPUT DIM" << endl;
    }
}
void setcolumn(vector< vector<double> > &c,double p){

    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    
    for(int i=0;i<m;i++){
        c[i][n-1] = p;
    }
}
void setrow(vector< vector<double> > &c,double p){
    unsigned long int m = c.size() ; unsigned long int n = c[0].size();
    for(int j=0;j<n;j++){
        c[m-1][j] = p;
    }
}
void equatMatrix(vector< vector<double> > &a,vector< vector<double> > &b){
     unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            b[i][j] = a[i][j];
        }
    }

    
    
                }
void hadamardproduct(vector< vector<double> > &c,vector< vector<double> > &a,vector< vector<double> > &b){
    unsigned long int m = a.size() ; unsigned long int n = a[0].size(); unsigned long int p = b.size() ; unsigned long int q = b[0].size();
    for(int i=0;i<m;i++){
        for(int j=0;j<q;j++){
            c[i][j] = ( a[i][j] ) * ( b[i][j] ) ;
        }
    }
    
}
void columnadd(vector< vector<double> > &c,vector< vector<double> > &a){
    unsigned long int m = a.size() ; unsigned long int n = a[0].size();
    double sum;
    for(int j=0;j<n;j++){
        double rollsum = 0;
        for(int i=0;i<m;i++){
            sum = a[i][j];
            rollsum = rollsum + sum;
        }
        c[0][j] = rollsum;
    }
    
}
void perError(int r,double &c ,vector< vector<double> > &a,vector< vector<double> > &b){
    unsigned long int m = a.size() ; unsigned long int n = a[0].size();
    double diff;
    double temp = 0;
    
    for(int i=0;i<m;i++){
        if(a[i][0] == 0){
            diff = 0;
        }
        else{
        diff = (double) ( abs( ( a[i][0] - b[i][0] ) / a[i][0] )  *  ( 100.00 ) );
        }
        temp = temp + diff;
    }
    c = temp/r;
}
void StandardDevY(double &c ,vector< vector<double> > &a){
    unsigned long int m = a.size();
    double mean;
    double run = 0;
    double sum;
    for(int i=0;i<m;i++){
        sum = a[i][0];
        run = run + sum;
    }
    mean = (double) run / m;
    run = 0;
    for(int i=0;i<m;i++){
        sum = ( a[i][0] - mean ) * ( a[i][0] - mean ) ;
        run = run + sum;
    }
    c = sqrt((double) run / m );
}
void errordiff(vector< vector<double> > &c,vector< vector<double> > &a,vector< vector<double> > &b){
    unsigned long int m = a.size();
    for(int i=0;i<m;i++){
        if(a[i][0] == 0){
            c[i][0] = 0;
        }
        else {
        c[i][0] = (abs( a[i][0] - b[i][0] ) / a[i][0] ) * 100.000 ;
        }
    }
}
void MaxCol(double &c ,vector< vector<double> > &a){
    unsigned long int m = a.size();
    double max = a[0][0];
    for(int i=1;i<m;i++){
        if(a[i][0] > max ){
            max = a[i][0];
        }
        else {
        }
    }
    c = max;
}
void testRun(int &a,int &b,int &c) {
    
    vector< vector<double> > x(b,vector<double>(a));
    vector< vector<double> > y(b,vector<double>(1));
    vector< vector<double> > y_bar(b,vector<double>(1));
    vector< vector<double> > w_one(a,vector<double>(c));
    vector< vector<double> > w_two(c,vector<double>(1));
    vector< vector<double> > z_two(b,vector<double>(c));
    vector< vector<double> > a_two(b,vector<double>(c));
    vector< vector<double> > z_three(b,vector<double>(1));
    double cost;

    textFileToMatrix(x,"InputVariables.txt");
    textFileToMatrix(y,"OutputVariables.txt");
    textFileToMatrix(w_one,"wone.txt");
    textFileToMatrix(w_two,"wtwo.txt");
    
    matrixmultiply(z_two,x,w_one);
    a_two = z_two;
    sigmoid(a_two);

    matrixmultiply(z_three,a_two,w_two);
    y_bar = z_three;
    sigmoid(y_bar);
    
    costfunc(cost,y_bar,y);

    printmatrix(w_one,"w_one",5);
    printmatrix(w_two,"w_two",5);
    printmatrix(x,"x",5);
    printmatrix(y,"y",15);
    printmatrix(y_bar,"y_bar",15);
    cout << "Cost = " << cost  << endl;
    MatrixtoTextFile(y_bar,"ybar.txt");
    
}
void batchOnlineRun(int &n,int &s,int it,int &exSize,int &descents,double &tar,int &rWone,int &rWtwo,bool &rw,double &nabla,double &mo){
    
    vector< vector<double> > x(exSize,vector<double>(n));
    vector< vector<double> > x_o(((exSize ) * ( it )),vector<double>(n));
    vector< vector<double> > x_transpose(n,vector<double>(exSize));
    vector< vector<double> > y(exSize,vector<double>(1));
    vector< vector<double> > y_o(((exSize ) * ( it )),vector<double>(1));
    vector< vector<double> > y_bar(exSize,vector<double>(1));
    vector< vector<double> > yError(exSize,vector<double>(1));
    vector< vector<double> > w_one(n,vector<double>(s));
    vector< vector<double> > djdw_one(n,vector<double>(s));
    vector< vector<double> > deltawone(n,vector<double>(s));
    vector< vector<double> > w_two(s,vector<double>(1));
    vector< vector<double> > djdw_two(s,vector<double>(1));
    vector< vector<double> > deltawtwo(s,vector<double>(1));
    vector< vector<double> > w_twoTranspose(1,vector<double>(s));
    vector< vector<double> > z_two(exSize,vector<double>(s));
    vector< vector<double> > z_twoprime(exSize,vector<double>(s));
    vector< vector<double> > a_two(exSize,vector<double>(s));
    vector< vector<double> > a_twoTranspose(s,vector<double>(exSize));
    vector< vector<double> > z_three(exSize,vector<double>(1));
    vector< vector<double> > z_threeprime(exSize,vector<double>(1));
    vector< vector<double> > delta_two(exSize,vector<double>(s));
    vector< vector<double> > delta_three(exSize,vector<double>(1));
    
    int    iter            = 0;
    int    count           = 0;
    double cost            = 100000;
    double percentageError = 100;
    double dev = 0;
    double maxdiff;
    
    textFileToMatrix(x_o,"InputVariables.txt");
    textFileToMatrix(y_o,"OutputVariables.txt");
    
    if(rw == true){
        setmatrixrandom(w_one,rWone);
        setmatrixrandom(w_two,rWtwo);
    }
    else {
        textFileToMatrix(w_one,"wone.txt");
        textFileToMatrix(w_two,"wtwo.txt");
    }
    double nill = 0.000;
    matrixScaler(deltawone,nill);
    matrixScaler(deltawtwo,nill);
    
    while( iter < it ){                                                                    // This will pass over each of the 30,000 lines of data
        
        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // Creates matrix x and y with 48 rows
        
        for(int i=0; i<exSize; i++){
            for(int j=0;j<n;j++){
                x[i][j] = x_o[( ( iter ) * ( exSize ) ) + i ][j];
            }
        }
        for(int i=0; i<exSize; i++){
            for(int j=0;j<1;j++){
                y[i][j] = y_o[( ( iter ) * ( exSize ) ) + i ][j];
            }
        }
        
        //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        matrixTranspose(x_transpose,x);

        while( percentageError > tar ){
            
            matrixmultiply(z_two,x,w_one);
            sigmoidAndPrime(z_two,a_two,z_twoprime);
            
            matrixTranspose(a_twoTranspose,a_two) ;
            matrixmultiply(z_three,a_two,w_two);
            sigmoidAndPrime(z_three,y_bar,z_threeprime);
            
            costfunc(cost,y_bar,y);
            perError(exSize,percentageError,y,y_bar);
            errordiff(yError,y,y_bar);
            //StandardDevY(dev,yError);
            MaxCol(maxdiff,yError);
            cout << count << " Cost = " << setprecision(20) << cost << "  Percentage Error = " << percentageError << "  Standard dev = " << dev << "  Max Error = " << maxdiff <<  "  iteration = " << iter << endl;
            
            matrixsubtract(delta_three,y_bar,y);
            hadamardproduct(delta_three,delta_three,z_threeprime);
            
            matrixmultiply(djdw_two,a_twoTranspose,delta_three);
            matrixTranspose(w_twoTranspose,w_two);
            
            matrixmultiply(delta_two,delta_three,w_twoTranspose);
            hadamardproduct(delta_two,delta_two,z_twoprime);
            matrixmultiply(djdw_one,x_transpose,delta_two);
            
            matrixScaler(djdw_one,nabla);
            matrixScaler(djdw_two,nabla);
            
            matrixScaler(deltawone,mo);
            matrixScaler(deltawtwo,mo);
            
            matrixsubtract(deltawone,deltawone,djdw_one);
            matrixsubtract(deltawtwo,deltawtwo,djdw_two);
            
            matrixadd(w_one,w_one,deltawone);
            matrixadd(w_two,w_two,deltawtwo);
            
            //matrixsubtract(w_one,w_one,djdw_one);
            //matrixsubtract(w_two,w_two,djdw_two);
       
            
            count ++ ;
            
            //------------------------------------------------------------
            
            // If its tried to fit the data with more than 1 million descents without hitting the target, it moves to the next line
            if(count > descents){
                cost = tar - 1.000 ;
            }
            //------------------------------------------------------------
            
        }
        
        if(percentageError < tar ){
            printmatrix(y,"y",15);
            printmatrix(y_bar,"y_bar",15);
            printmatrix(w_one,"w_one",5);
            printmatrix(w_two,"w_two",5);
            printmatrix(x,"x",5);
            MatrixtoTextFile(w_one,"wone.txt");
            MatrixtoTextFile(w_two,"wtwo.txt");
            MatrixtoTextFile(y_bar,"ybar.txt");
            iter++;
            cost            = 0;
            count           = 0;
            percentageError = 100;
            
        }
    }
}
void offlineRun(int &n,int &dc,int &s,double &tar,int &rWone,int &rWtwo,bool &rw){
    
    vector< vector<double> > x(dc,vector<double>(n));
    vector< vector<double> > x_transpose(n,vector<double>(dc));
    vector< vector<double> > y(dc,vector<double>(1));
    vector< vector<double> > y_bar(dc,vector<double>(1));
    vector< vector<double> > yError(dc,vector<double>(1));
    vector< vector<double> > w_one(n,vector<double>(s));
    vector< vector<double> > djdw_one(n,vector<double>(s));
    vector< vector<double> > w_two(s,vector<double>(1));
    vector< vector<double> > djdw_two(s,vector<double>(1));
    vector< vector<double> > w_twoTranspose(1,vector<double>(s));
    vector< vector<double> > z_two(dc,vector<double>(s));
    vector< vector<double> > z_twoprime(dc,vector<double>(s));
    vector< vector<double> > a_two(dc,vector<double>(s));
    vector< vector<double> > a_twoTranspose(s,vector<double>(dc));
    vector< vector<double> > z_three(dc,vector<double>(1));
    vector< vector<double> > z_threeprime(dc,vector<double>(1));
    vector< vector<double> > delta_two(dc,vector<double>(s));
    vector< vector<double> > delta_three(dc,vector<double>(1));
    int iter    = 0;
    int count   = 0;
    double cost = 1;
    double percentageError = 100;
    double dev;
    double maxdiff;
    
    textFileToMatrix(x,"InputVariables.txt");
    textFileToMatrix(y,"OutputVariables.txt");
    
    if(rw == true){
        setmatrixrandom(w_one,rWone);
        setmatrixrandom(w_two,rWtwo);
    }
    else {
        textFileToMatrix(w_one,"wone.txt");
        textFileToMatrix(w_two,"wtwo.txt");
    }

    matrixTranspose(x_transpose,x);
        
    while( percentageError > tar ){
            
            matrixmultiply(z_two,x,w_one);
            sigmoidAndPrime(z_two,a_two,z_twoprime);
            
            matrixTranspose(a_twoTranspose,a_two) ;
            matrixmultiply(z_three,a_two,w_two);
            sigmoidAndPrime(z_three,y_bar,z_threeprime);
            
            costfunc(cost,y_bar,y);
            perError(dc,percentageError,y,y_bar);
            errordiff(yError,y,y_bar);
            StandardDevY(dev,yError);
            MaxCol(maxdiff,yError);
            cout << count << " Cost = " << setprecision(20) << cost << "  Percentage Error = " << percentageError << "  Standard dev = " << dev << "  Max Error = " << maxdiff <<  "  iteration = " << iter << endl;
            
            matrixsubtract(delta_three,y_bar,y);
            hadamardproduct(delta_three,delta_three,z_threeprime);
            
            matrixmultiply(djdw_two,a_twoTranspose,delta_three);
            matrixTranspose(w_twoTranspose,w_two);
            
            matrixmultiply(delta_two,delta_three,w_twoTranspose);
            hadamardproduct(delta_two,delta_two,z_twoprime);
            matrixmultiply(djdw_one,x_transpose,delta_two);
            
            matrixsubtract(w_one,w_one,djdw_one);
            matrixsubtract(w_two,w_two,djdw_two);
            
            count ++ ;
        }
        
        if(percentageError < tar ){
            printmatrix(y,"y",15);
            printmatrix(y_bar,"y_bar",15);
            printmatrix(w_one,"w_one",5);
            printmatrix(w_two,"w_two",5);
            printmatrix(x,"x",5);
            MatrixtoTextFile(w_one,"wone.txt");
            MatrixtoTextFile(w_two,"wtwo.txt");
            }
}

    
