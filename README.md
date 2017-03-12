# simpleneuralnetwork

(1) In the text files "InputVariables.txt" and "OutputVaribles.txt" copy your input data and your output data into these, space deliminated.

(2) Open the file "main.cpp" in any IDE, make sure it's running C++11 or higher. I wrote the code in Xcode on my Mac so it might be useful to use that as your IDE.

(3) In the main() function, there are several boolean variables, set them to ;

    bool batchOnline = true;
    bool offline     = false;
    bool test        = false;

The main method we will use is the "BatchOnline" which means we can vary the size of the input data X, so instead of using 40,000 examples at once, we will start with around 100 and work out way upto larger numbers if neccisary. This will converge much faster although it might not be a sufficent minima.

Under the heading // Overall Setup change the values as required. The line "int layers =3" doesn't do anything so there is no need to worry about this.

    int    numberOfVariables       = 11;
    This is the number of variables in the input data
    
    int    layers                  = 3;
    Ignore this
    
    int    nodesInFirstHiddenLayer = 16;
    This is how many nodes are in the hidden layer, this number can be any integer, there is no rule of thumb for what value this should be. However if you have too many nodes it will increases computing time and may begin to model noise in your data.
    
    int    rangeWone               = 4;
    int    rangeWtwo               = 4;
    This is the range of values for the initial random weightings, a value of 4 will give random values between -4 and 4.
    
    bool   randomiseWeights        = false;
    Set this to true initially until we find a minima to pause the programme at. 
    
    double percentageErrorTarget   = 3.9;
    This is the target error we want. I'd set this high initially. Remember we can run the programme over the data several times and the weightings W1 and W2 are autosaved whenever you hit the percentage error target.
    
    double learningRate            = 0.0001;
    Set this value beween 0 and 1, this will take some guesses, but you will know when you have a good value as the cost wont jump around so chaotically.
    
    double momentum                = 0.75;
    Set this value between 0 and 1. If the learning rate is close to one, then the momentum in general should be lowered. This again takes some playing around with.
    
    // Batch setup
    int iterations            = 1;
    If you are going to do a small subset of the data, or groups of data, you may want to run over the data several times.
    
    int exampleSize           = 13853;
    This is the number of examples
    
    int numberOfDescents      = 1000000;
    This is the number of weight updates until the programme stops and autosaves the values for W1 and W2. This is useful if you want to run the programme for long periods of time.
    
    int times                 = 1;
    This is useful when using groups of data.
    
     // Testing setup
    int xColumns              = numberOfVariables;
    Leave this.
    
    int xRows                 = 13853;
    When are happy with our values for W1 and W2 and want to run our programme on fresh data, this is the number of examples.
    
    int wOneColumns           = nodesInFirstHiddenLayer;
    Leave this.
    
   (4) Now compile and run the programme. Do this until you're happy with the cost/percentage error. 
    
   (5) Now we want to use our W1 and W2 we calculated on fresh data. Again in the text files "InputVariables.txt" and "OutputVaribles.txt" copy your input data and your output data into these, space deliminated.
    
   (6) Set the following 
    
        bool batchOnline = false;
        bool offline     = false;
        bool test        = true;
    
    (7) Scroll down to this
        ```
        // Testing setup
        int xRows                 = 13853;
        Set this to the number of examples you have of fresh data.
        ```
    (8) Compile and run the programme. 
    
    (9) In the programmes products directory there will now be a file named "ybar.txt" this file will contain the estimates for your output values. You can now compare this to the actual outputvalues. 
    
    
    
