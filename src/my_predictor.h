// V2: Multi-perspective Perceptron (~8MB): 4.016
#include <cstring>
#include <bitset>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <algorithm>

#define TABLE_SIZE 32768
#define HIST_LEN 60
#define INITIAL_THRESHOLD 60
#define THRESHOLD_MAX 100
#define THRESHOLD_MIN 20

class my_update : public branch_update {
public:
    unsigned int index;
    int weighted_sum;
};

class my_predictor : public branch_predictor {
public:
    my_update u;
    branch_info bi;
    std::bitset<HIST_LEN> global_history;           // Global history register
    int perceptronTable[TABLE_SIZE][HIST_LEN + 1];  // Perceptron table: 60 Neuron weights + 1 bias weight
    int pathHistory[TABLE_SIZE];                    // Path history table: 32,768 addresses
    int recencyStack[16];                           // Recency stack:  16 most recent branch addresses
    int correctPredictionCounter;                   // # of consecutive correct predictions
    int dynamicThreshold;                           // Determine prediction confidence

    my_predictor(void) : correctPredictionCounter(0), dynamicThreshold(INITIAL_THRESHOLD) 
    {
        global_history.reset();
        memset(perceptronTable, 0, sizeof(perceptronTable));
        memset(pathHistory, 0, sizeof(pathHistory));
        memset(recencyStack, -1, sizeof(recencyStack));
    }

    branch_update *predict(branch_info &b) 
    {
        bi = b;
        if (b.br_flags & BR_CONDITIONAL) 
        {
            // Find perceptron index
            u.index = hashPC(b.address);

            // weighted sum = bias weight
            int x = perceptronTable[u.index][0];

            // weighted sum + global history
            for (int i = 0; i < HIST_LEN; i++)
                x += perceptronTable[u.index][i + 1] * (global_history[i] ? 1 : -1);

            // weighted sum + (1 if address even else 0)
            x += pathHistory[u.index] * (b.address % 2 == 0 ? 1 : -1);

            // weighted sum + (constant value if branch was recently seen)
            for (int i = 0; i < 16; i++) 
            {
                if (recencyStack[i] == static_cast<int>(b.address)) 
                {
                    x += 10;
                    break;
                }
            }

            // Store the weighted sum
            u.weighted_sum = x;

            // Taken if weighted sum not negative
            u.direction_prediction(x >= 0);
        } 
        else 
        {
            u.direction_prediction(true);
        }
        u.target_prediction(0);
        return &u;
    }

    void update(branch_update *u, bool taken, unsigned int target) 
    {
        if (bi.br_flags & BR_CONDITIONAL) 
        {
            int index = ((my_update *)u)->index;
            int x = ((my_update *)u)->weighted_sum;

            // If prediction incorrect or confidence low
            if ((x >= 0) != taken || abs(x) <= dynamicThreshold) 
            {
                // Increase learning rate if confidence is low
                int updateValue = (abs(x) <= dynamicThreshold / 2) ? 2 : 1;

                // Update bias weight
                perceptronTable[index][0] += taken ? updateValue : -updateValue;
                // Saturation: constrain to [-dynamicThreshold, dynamicThreshold]
                perceptronTable[index][0] = std::max(-dynamicThreshold, std::min(dynamicThreshold, perceptronTable[index][0]));

                // Update neuron weights
                for (int i = 0; i < HIST_LEN; i++) 
                {
                    if (global_history[i] == taken)
                        perceptronTable[index][i + 1] += updateValue;
                    else
                        perceptronTable[index][i + 1] -= updateValue;
                    perceptronTable[index][i + 1] = std::max(-dynamicThreshold, std::min(dynamicThreshold, perceptronTable[index][i + 1]));
                }

                // Update path history
                pathHistory[index] += taken ? 1 : -1;
                pathHistory[index] = std::max(-dynamicThreshold, std::min(dynamicThreshold, pathHistory[index]));

                // Reset consecutive correct prediction counter
                correctPredictionCounter = 0;
            } 
            else // If correct prediction
            {
                // Increment consecutive correct prediction counter
                correctPredictionCounter++;

                // Increase threshold if more than 10 consecutive correct predictions (and reset prediction counter)
                if (correctPredictionCounter > 10) 
                {
                    dynamicThreshold = std::min(dynamicThreshold + 1, THRESHOLD_MAX);
                    correctPredictionCounter = 0;
                }
            }

            // Push current address onto recency stack
            for (int i = 15; i > 0; i--)
                recencyStack[i] = recencyStack[i - 1];
            recencyStack[0] = bi.address;

            // Update global history: left shift and set LSB to 1 (T) or 0 (NT)
            global_history <<= 1;
            global_history.set(0, taken);
        }
    }

private:
    unsigned int hashPC(unsigned int pc) {
        return (pc ^ (global_history.to_ulong() << 1) ^ (pc >> 2)) % TABLE_SIZE;
    }
};