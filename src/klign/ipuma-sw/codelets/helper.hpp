#ifndef CODELET_SHARED_HPP
#define CODELET_SHARED_HPP

const short GAP_PENALTY = -1;

inline short compareElements(char a, char b) {
    if (a == b)
        return 1;
    return -1;
}

/**
 * Calculate a single SW cell.
 */
inline void calcCell(short top, short left, short diag, short* sScore, char* sDir) {
    short max = 0;
    short maxDir = 'n';

    if (top > max && top > left && top > diag) {
        max = top;
        maxDir = 't';
    } else if (left > diag && left > max) {
        max = left;
        maxDir = 'l';
    } else if (diag > max) {
        max = diag;
        maxDir = 'd';
    }

    *sScore = max;
    *sDir = maxDir;
}

/**
 * Calculate a single SW cell.
 */
inline void calcCell(int top, int left, int diag, int* sScore, char* sDir) {
    int max = 0;
    char maxDir = 'n';

    if (top > max && top > left && top > diag) {
        max = top;
        maxDir = 't';
    } else if (left > diag && left > max) {
        max = left;
        maxDir = 'l';
    } else if (diag > max) {
        max = diag;
        maxDir = 'd';
    }

    *sScore = max;
    *sDir = maxDir;
}

#endif // CODELET_SHARED_HPP