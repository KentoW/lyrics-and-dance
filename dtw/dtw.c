#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_NUM_COUNT 16000     // number of motion codebooks + number of lyrics codebooks
#define MAX_NUM_VALUE 16000     // number of motion codebooks + number of lyrics codebooks
#define DEFAULT_COST 0.5        // default cost
#define MAX_DATABASE_LINES 100000

double cost_matrix[MAX_NUM_VALUE + 1][MAX_NUM_VALUE + 1];

typedef struct {
    int *query;
    int query_len;
    int **database;
    int *database_lens;
    int start;
    int end;
} ThreadData;

void load_costs(const char *filename) {
    for (int i = 0; i <= MAX_NUM_VALUE; i++) {
        for (int j = 0; j <= MAX_NUM_VALUE; j++) {
            cost_matrix[i][j] = (i == j) ? 0 : DEFAULT_COST;
        }
    }
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open cost file");
        exit(EXIT_FAILURE);
    }
    int a, b;
    double cost;
    while (fscanf(file, "%d %d %lf", &a, &b, &cost) == 3) {
        cost_matrix[a][b] = cost;
        cost_matrix[b][a] = cost;
    }
    fclose(file);
}

double calculate_edit_distance(int *s1, int len1, int *s2, int len2) {
    double dist[len1 + 1][len2 + 1];
    for (int i = 0; i <= len1; i++) dist[i][0] = i;
    for (int j = 0; j <= len2; j++) dist[0][j] = j;
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            double cost = cost_matrix[s1[i - 1]][s2[j - 1]];
            double min = dist[i - 1][j - 1] + cost;
            // insertion and deletion cost assign a fixed cost of 1.
            if (dist[i - 1][j] + 1 < min) min = dist[i - 1][j] + 1;
            if (dist[i][j - 1] + 1 < min) min = dist[i][j - 1] + 1;
            dist[i][j] = min;
        }
    }
    return dist[len1][len2];
}

void *process_database(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; i++) {
        double dist = calculate_edit_distance(data->query, data->query_len, data->database[i], data->database_lens[i]);
        printf("%d:%f\n", i, dist);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <query.txt> <database.txt> <cost.txt> <num_threads>\n", argv[0]);
        return 1;
    }

    char *query_filename = argv[1];
    char *database_filename = argv[2];
    char *cost_filename = argv[3];
    int num_threads = atoi(argv[4]);

    load_costs(cost_filename);

    FILE *file = fopen(query_filename, "r");
    if (!file) {
        perror("Failed to open query file");
        exit(EXIT_FAILURE);
    }
    int query[MAX_NUM_COUNT], query_len = 0;
    while (fscanf(file, "%d", &query[query_len]) == 1) {
        query_len++;
    }
    fclose(file);

    file = fopen(database_filename, "r");
    if (!file) {
        perror("Failed to open database file");
        exit(EXIT_FAILURE);
    }
    int **database = malloc(MAX_DATABASE_LINES * sizeof(int *));
    int *database_lens = malloc(MAX_DATABASE_LINES * sizeof(int));
    if (!database || !database_lens) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    char line[4096];
    int db_size = 0;
    while (fgets(line, sizeof(line), file) && db_size < MAX_DATABASE_LINES) {
        char *token;
        int count = 0;
        database[db_size] = malloc(MAX_NUM_COUNT * sizeof(int));
        if (!database[db_size]) {
            fprintf(stderr, "Memory allocation failed for database line %d\n", db_size);
            exit(EXIT_FAILURE);
        }
        token = strtok(line, " ");
        while (token != NULL) {
            database[db_size][count++] = atoi(token);
            token = strtok(NULL, " ");
        }
        database_lens[db_size++] = count;
    }
    fclose(file);

    pthread_t threads[num_threads];
    ThreadData data[num_threads];
    int chunk_size = db_size / num_threads;
    for (int i = 0; i < num_threads; i++) {
        data[i].query = query;
        data[i].query_len = query_len;
        data[i].database = database;
        data[i].database_lens = database_lens;
        data[i].start = i * chunk_size;
        data[i].end = (i == num_threads - 1) ? db_size : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, process_database, &data[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    for (int i = 0; i < db_size; i++) {
        free(database[i]);
    }
    return 0;
}

