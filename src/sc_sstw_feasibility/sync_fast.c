#include <math.h>
#include <stdlib.h>

static double sc_sstw_best_possible_average_cost_lower_bound(
    const double *costs,
    const int *counts,
    int cols,
    int rows,
    int processed_rows,
    double skip_penalty
) {
    int remaining_rows = rows - processed_rows;
    double lower_bound = INFINITY;
    for (int col = 0; col <= cols; ++col) {
        int required_skip_count = cols - col - remaining_rows;
        if (required_skip_count < 0) {
            required_skip_count = 0;
        }
        double minimum_final_cost = costs[col] + (double)required_skip_count * skip_penalty;
        int remaining_columns = cols - col;
        int future_match_count = remaining_rows < remaining_columns ? remaining_rows : remaining_columns;
        int maximum_final_match_count = counts[col] + future_match_count;
        if (maximum_final_match_count < 1) {
            maximum_final_match_count = 1;
        }
        double average_lower_bound = minimum_final_cost / (double)maximum_final_match_count;
        if (average_lower_bound < lower_bound) {
            lower_bound = average_lower_bound;
        }
    }
    return lower_bound;
}

int sc_sstw_dynamic_time_sync_score_bounded_flat(
    const double *observed_xy,
    int rows,
    const double *candidate_xy,
    int cols,
    double min_score_to_beat,
    double skip_penalty,
    double repeat_penalty,
    double *out_score,
    int *out_abandoned
) {
    double *previous_costs = (double *)malloc((size_t)(cols + 1) * sizeof(double));
    double *current_costs = (double *)malloc((size_t)(cols + 1) * sizeof(double));
    int *previous_counts = (int *)malloc((size_t)(cols + 1) * sizeof(int));
    int *current_counts = (int *)malloc((size_t)(cols + 1) * sizeof(int));
    if (!previous_costs || !current_costs || !previous_counts || !current_counts) {
        free(previous_costs);
        free(current_costs);
        free(previous_counts);
        free(current_counts);
        return 1;
    }

    *out_abandoned = 0;
    double best_average_cost_to_beat = -min_score_to_beat;
    int use_bound = !isinf(min_score_to_beat);
    if (rows == 0) {
        use_bound = 0;
    }

    previous_costs[0] = 0.0;
    previous_counts[0] = 0;
    for (int col = 1; col <= cols; ++col) {
        previous_costs[col] = previous_costs[col - 1] + skip_penalty;
        previous_counts[col] = previous_counts[col - 1];
    }
    if (use_bound) {
        double lower_bound = sc_sstw_best_possible_average_cost_lower_bound(
            previous_costs,
            previous_counts,
            cols,
            rows,
            0,
            skip_penalty
        );
        if (lower_bound > best_average_cost_to_beat) {
            *out_score = min_score_to_beat;
            *out_abandoned = 1;
            free(previous_costs);
            free(current_costs);
            free(previous_counts);
            free(current_counts);
            return 0;
        }
    }

    for (int row = 0; row < rows; ++row) {
        double observed_x = observed_xy[2 * row];
        double observed_y = observed_xy[2 * row + 1];
        current_costs[0] = INFINITY;
        current_counts[0] = 0;
        for (int col = 1; col <= cols; ++col) {
            double delta_x = observed_x - candidate_xy[2 * (col - 1)];
            double delta_y = observed_y - candidate_xy[2 * (col - 1) + 1];
            double match_cost = delta_x * delta_x + delta_y * delta_y;

            double best_cost = previous_costs[col - 1] + match_cost;
            int best_count = previous_counts[col - 1] + 1;

            double skip_cost = current_costs[col - 1] + skip_penalty;
            if (skip_cost < best_cost) {
                best_cost = skip_cost;
                best_count = current_counts[col - 1];
            }

            double repeat_cost = previous_costs[col] + repeat_penalty + match_cost;
            if (repeat_cost < best_cost) {
                best_cost = repeat_cost;
                best_count = previous_counts[col];
            }

            current_costs[col] = best_cost;
            current_counts[col] = best_count;
        }

        double *tmp_costs = previous_costs;
        previous_costs = current_costs;
        current_costs = tmp_costs;
        int *tmp_counts = previous_counts;
        previous_counts = current_counts;
        current_counts = tmp_counts;

        if (use_bound) {
            double lower_bound = sc_sstw_best_possible_average_cost_lower_bound(
                previous_costs,
                previous_counts,
                cols,
                rows,
                row + 1,
                skip_penalty
            );
            if (lower_bound > best_average_cost_to_beat) {
                *out_score = min_score_to_beat;
                *out_abandoned = 1;
                free(previous_costs);
                free(current_costs);
                free(previous_counts);
                free(current_counts);
                return 0;
            }
        }
    }

    int match_count = previous_counts[cols] > 0 ? previous_counts[cols] : 1;
    double average_cost = previous_costs[cols] / (double)match_count;
    *out_score = -average_cost;
    free(previous_costs);
    free(current_costs);
    free(previous_counts);
    free(current_counts);
    return 0;
}

int sc_sstw_dynamic_time_sync_score_bounded_flat_workspace(
    const double *observed_xy,
    int rows,
    const double *candidate_xy,
    int cols,
    double min_score_to_beat,
    double skip_penalty,
    double repeat_penalty,
    double *previous_costs,
    double *current_costs,
    int *previous_counts,
    int *current_counts,
    double *out_score,
    int *out_abandoned
) {
    *out_abandoned = 0;
    double best_average_cost_to_beat = -min_score_to_beat;
    int use_bound = !isinf(min_score_to_beat);
    if (rows == 0) {
        use_bound = 0;
    }

    previous_costs[0] = 0.0;
    previous_counts[0] = 0;
    for (int col = 1; col <= cols; ++col) {
        previous_costs[col] = previous_costs[col - 1] + skip_penalty;
        previous_counts[col] = previous_counts[col - 1];
    }
    if (use_bound) {
        double lower_bound = sc_sstw_best_possible_average_cost_lower_bound(
            previous_costs,
            previous_counts,
            cols,
            rows,
            0,
            skip_penalty
        );
        if (lower_bound > best_average_cost_to_beat) {
            *out_score = min_score_to_beat;
            *out_abandoned = 1;
            return 0;
        }
    }

    for (int row = 0; row < rows; ++row) {
        double observed_x = observed_xy[2 * row];
        double observed_y = observed_xy[2 * row + 1];
        current_costs[0] = INFINITY;
        current_counts[0] = 0;
        for (int col = 1; col <= cols; ++col) {
            double delta_x = observed_x - candidate_xy[2 * (col - 1)];
            double delta_y = observed_y - candidate_xy[2 * (col - 1) + 1];
            double match_cost = delta_x * delta_x + delta_y * delta_y;

            double best_cost = previous_costs[col - 1] + match_cost;
            int best_count = previous_counts[col - 1] + 1;

            double skip_cost = current_costs[col - 1] + skip_penalty;
            if (skip_cost < best_cost) {
                best_cost = skip_cost;
                best_count = current_counts[col - 1];
            }

            double repeat_cost = previous_costs[col] + repeat_penalty + match_cost;
            if (repeat_cost < best_cost) {
                best_cost = repeat_cost;
                best_count = previous_counts[col];
            }

            current_costs[col] = best_cost;
            current_counts[col] = best_count;
        }

        double *tmp_costs = previous_costs;
        previous_costs = current_costs;
        current_costs = tmp_costs;
        int *tmp_counts = previous_counts;
        previous_counts = current_counts;
        current_counts = tmp_counts;

        if (use_bound) {
            double lower_bound = sc_sstw_best_possible_average_cost_lower_bound(
                previous_costs,
                previous_counts,
                cols,
                rows,
                row + 1,
                skip_penalty
            );
            if (lower_bound > best_average_cost_to_beat) {
                *out_score = min_score_to_beat;
                *out_abandoned = 1;
                return 0;
            }
        }
    }

    int match_count = previous_counts[cols] > 0 ? previous_counts[cols] : 1;
    double average_cost = previous_costs[cols] / (double)match_count;
    *out_score = -average_cost;
    return 0;
}

int sc_sstw_score_candidates_margin_proof_workspace(
    const double *observed_xy,
    int rows,
    const double *candidate_xy_matrix,
    const int *candidate_roles,
    int candidate_count,
    int cols,
    double diagnostic_margin,
    double skip_penalty,
    double repeat_penalty,
    double *previous_costs,
    double *current_costs,
    int *previous_counts,
    int *current_counts,
    int *out_best_owner_index,
    double *out_best_owner_score,
    int *out_best_wrong_index,
    double *out_best_wrong_score,
    int *out_scored_count,
    int *out_abandoned_count
) {
    int best_owner_index = -1;
    double best_owner_score = -INFINITY;
    int best_wrong_index = -1;
    double best_wrong_score = -INFINITY;
    int scored_count = 0;
    int abandoned_count = 0;

    for (int index = 0; index < candidate_count; ++index) {
        if (candidate_roles[index] != 0) {
            continue;
        }
        double score = 0.0;
        int abandoned = 0;
        double min_score_to_beat = best_owner_index >= 0 ? best_owner_score : -INFINITY;
        int result = sc_sstw_dynamic_time_sync_score_bounded_flat_workspace(
            observed_xy,
            rows,
            candidate_xy_matrix + (size_t)index * (size_t)cols * 2U,
            cols,
            min_score_to_beat,
            skip_penalty,
            repeat_penalty,
            previous_costs,
            current_costs,
            previous_counts,
            current_counts,
            &score,
            &abandoned
        );
        if (result != 0) {
            return result;
        }
        scored_count += 1;
        abandoned_count += abandoned;
        if (!abandoned && (best_owner_index < 0 || score > best_owner_score)) {
            best_owner_index = index;
            best_owner_score = score;
        }
    }

    if (best_owner_index < 0) {
        return 2;
    }

    double proof_threshold = best_owner_score - diagnostic_margin;
    for (int index = 0; index < candidate_count; ++index) {
        if (candidate_roles[index] != 1) {
            continue;
        }
        double score = 0.0;
        int abandoned = 0;
        double min_score_to_beat = best_wrong_index >= 0 && best_wrong_score > proof_threshold
            ? best_wrong_score
            : proof_threshold;
        int result = sc_sstw_dynamic_time_sync_score_bounded_flat_workspace(
            observed_xy,
            rows,
            candidate_xy_matrix + (size_t)index * (size_t)cols * 2U,
            cols,
            min_score_to_beat,
            skip_penalty,
            repeat_penalty,
            previous_costs,
            current_costs,
            previous_counts,
            current_counts,
            &score,
            &abandoned
        );
        if (result != 0) {
            return result;
        }
        scored_count += 1;
        abandoned_count += abandoned;
        if (!abandoned && (best_wrong_index < 0 || score > best_wrong_score)) {
            best_wrong_index = index;
            best_wrong_score = score;
        }
    }

    *out_best_owner_index = best_owner_index;
    *out_best_owner_score = best_owner_score;
    if (best_wrong_index < 0 || best_wrong_score <= proof_threshold) {
        *out_best_wrong_index = -1;
        *out_best_wrong_score = proof_threshold;
    } else {
        *out_best_wrong_index = best_wrong_index;
        *out_best_wrong_score = best_wrong_score;
    }
    *out_scored_count = scored_count;
    *out_abandoned_count = abandoned_count;
    return 0;
}
