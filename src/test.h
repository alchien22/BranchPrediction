// my_predictor.h
// This file contains a modified my_predictor class.
// We enhance the baseline gshare to a simplified MTAGE + SC-based predictor.
#define HISTORY_LENGTH 15
#define TABLE_BITS 15
#define TAGE_TABLES 4
#define SC_BITS 6

class my_update : public branch_update {
public:
    unsigned int index;
    unsigned int tage_index[TAGE_TABLES];
    bool tage_prediction;
    bool sc_prediction;
};

class my_predictor : public branch_predictor {
public:
    my_update u;
    branch_info bi;
    unsigned int history;
    unsigned char gshare_tab[1 << TABLE_BITS];
    unsigned char tage_tables[TAGE_TABLES][1 << TABLE_BITS];
    signed char sc_table[1 << SC_BITS];

    my_predictor(void) : history(0) {
        memset(gshare_tab, 0, sizeof(gshare_tab));
        memset(tage_tables, 0, sizeof(tage_tables));
        memset(sc_table, 0, sizeof(sc_table));
    }

    branch_update *predict(branch_info &b) {
        bi = b;
        if (b.br_flags & BR_CONDITIONAL) {
            // Gshare prediction
            u.index = (history << (TABLE_BITS - HISTORY_LENGTH)) ^ (b.address & ((1 << TABLE_BITS) - 1));
            bool gshare_prediction = (gshare_tab[u.index] >> 1);

            // TAGE-like prediction - using multiple tables
            u.tage_prediction = false;
            for (int i = 0; i < TAGE_TABLES; i++) {
                u.tage_index[i] = ((b.address >> i) ^ (history)) & ((1 << TABLE_BITS) - 1);
                if (tage_tables[i][u.tage_index[i]] > 1) {
                    u.tage_prediction = true;
                    break;
                }
            }

            // Statistical Corrector (SC) prediction
            int sc_index = (b.address ^ history) & ((1 << SC_BITS) - 1);
            u.sc_prediction = sc_table[sc_index] >= 0;

            // Final prediction based on majority vote of Gshare, TAGE, and SC
            int vote = (gshare_prediction ? 1 : -1) + (u.tage_prediction ? 1 : -1) + (u.sc_prediction ? 1 : -1);
            u.direction_prediction(vote > 0);
        } else {
            u.direction_prediction(true);
        }
        u.target_prediction(0);
        return &u;
    }

    void update(branch_update *u, bool taken, unsigned int target) {
        if (bi.br_flags & BR_CONDITIONAL) {
            // Update Gshare
            unsigned char *g = &gshare_tab[((my_update *)u)->index];
            if (taken) {
                if (*g < 3)
                    (*g)++;
            } else {
                if (*g > 0)
                    (*g)--;
            }

            // Update TAGE tables
            for (int i = 0; i < TAGE_TABLES; i++) {
                unsigned char *t = &tage_tables[i][((my_update *)u)->tage_index[i]];
                if (taken) {
                    if (*t < 3)
                        (*t)++;
                } else {
                    if (*t > 0)
                        (*t)--;
                }
            }

            // Update Statistical Corrector (SC) table
            int sc_index = (bi.address ^ history) & ((1 << SC_BITS) - 1);
            if (taken) {
                if (sc_table[sc_index] < 127)
                    sc_table[sc_index]++;
            } else {
                if (sc_table[sc_index] > -128)
                    sc_table[sc_index]--;
            }

            // Update history
            history <<= 1;
            history |= taken;
            history &= (1 << HISTORY_LENGTH) - 1;
        }
    }
};
