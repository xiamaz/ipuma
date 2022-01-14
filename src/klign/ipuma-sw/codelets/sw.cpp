#include <poplar/Vertex.hpp>
#include <poplar/FieldTypes.hpp>
#include <print.h>

#include <poplar/StackSizeDefs.hpp>
#include "poplar/AvailableVTypes.h"

#include "helper.hpp"


// const short GAP_PENALTY = -1;
// 
// short compareElements(char a, char b) {
//     if (a == b)
//         return 1;
//     return -1;
// }
// 
// /**
//  * Calculate a single SW cell.
//  */
// void calcCell(short top, short left, short diag, short* sScore, char* sDir) {
//     short max = 0;
//     short maxDir = 'n';
// 
//     if (top > max && top > left && top > diag) {
//         max = top;
//         maxDir = 't';
//     } else if (left > diag && left > max) {
//         max = left;
//         maxDir = 'l';
//     } else if (diag > max) {
//         max = diag;
//         maxDir = 'd';
//     }
// 
//     *sScore = max;
//     *sDir = maxDir;
// }

#include "print.h"

class Testoo : public poplar::Vertex {
public:

bool compute() {
        printf("Hello form IPU\n");
        printf("Hello form IPU\n");
        return true;
}

};

/**
 * Single SW operations for cell i, j
 */
class SWOperation : public poplar::Vertex {
public:
    // Fields
    poplar::Input<char> ref;
    poplar::Input<char> query;
    poplar::Input<short> topScore;
    poplar::Input<short> leftScore;
    poplar::Input<short> diagScore;
    poplar::Output<char> dir;
    poplar::Output<short> score;

    bool compute() {
        short top = topScore + GAP_PENALTY;
        short left = leftScore + GAP_PENALTY;
        short diag = diagScore + compareElements(ref, query);

        short sScore = 0;
        char sDir = 'n';

        calcCell(top, left, diag, &sScore, &sDir);

        *dir = sDir;
        *score = sScore;
        return true;
    }
};


/**
 * Chunked SW operations for cell i, j
 */
class SWOperationChunk : public poplar::Vertex {
public:
    // Fields
    poplar::Input<poplar::Vector<char>> ref;
    poplar::Input<poplar::Vector<char>> query;
    poplar::Input<poplar::Vector<short>> topScores;
    poplar::Input<poplar::Vector<short>> leftScores;
    poplar::Input<short> diagScoreInitial;

    poplar::Output<poplar::VectorList<char, poplar::VectorListLayout::DELTANELEMENTS>> dir;
    poplar::InOut<poplar::VectorList<short, poplar::VectorListLayout::DELTANELEMENTS>> score;

    bool compute() {
        size_t m = ref.size();
        size_t n = query.size();

        int maxE = m + n;
        for (int e = 0; e <= maxE - 2; ++e) {
            // printf("Hello world %d %d %d\n", maxE, m, n);
            for (int i = 0, j = e - i; i < m && j >= 0; ++i, --j) {

                if (j >= n) {
                    size_t diff = j - n + 1;
                    j -= diff;
                    i += diff;
                }
                short topScore, leftScore, diagScore;
                if (j == 0) {
                    leftScore = leftScores[i];
                } else {
                    leftScore = score[i][j-1];
                }
                if (i == 0) {
                    topScore = topScores[j];
                } else {
                    topScore = score[i-1][j];
                }
                if (i == 0 && j == 0) {
                    diagScore = diagScoreInitial;
                } else {
                    // find diag score among top/left or inside
                    if (i > 0 && j > 0) {
                        diagScore = score[i-1][j-1];
                    } else if (i == 0 && j > 0) {
                        diagScore = topScores[j-1];
                    } else if (i > 0 && j == 0) {
                        diagScore = leftScores[i-1];
                    }
                }

                short top = topScore + GAP_PENALTY;
                short left = leftScore + GAP_PENALTY;
                short diag = diagScore + compareElements(ref[i], query[j]);
                short sScore = 0;
                char sDir = 'n';

                calcCell(top, left, diag, &sScore, &sDir);

                // printf("SW Codelet ([%d]%c:[%d]%c) %d %c\n", i, ref[i], j, query[j], sScore, sDir);
                // printf("T: %d L: %d D: %d(%d) S: %d\n", top, left, diag, diagScore, sScore);

                score[i][j] = sScore;
                dir[i][j] = sDir;
            }
        }

        return true;
    }
};