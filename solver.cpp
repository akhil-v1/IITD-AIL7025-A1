#include "solver.h"
#include <iostream>
#include <chrono>
#include <limits>
#include <algorithm>
#include <vector>
#include <random>

using namespace std;

const int DRY = 0, PER = 1, OTH = 2;


double calculateVillageValue(const Village& village, int dry_delivered, int perishable_delivered, int other_delivered,const vector<PackageInfo>& packages) {
    int max_food_needed = 9 * village.population;
    int max_other_needed = village.population;
    
    int total_food_delivered = dry_delivered + perishable_delivered;
    int effective_food = min(total_food_delivered, max_food_needed);
    int effective_other = min(other_delivered, max_other_needed);
    
    double food_value = 0;
    if (effective_food > 0) {
        int effective_perishable = min(perishable_delivered, effective_food);
        food_value += effective_perishable * packages[PER].value;
        
        int remaining_food_need = effective_food - effective_perishable;
        int effective_dry = min(dry_delivered, remaining_food_need);
        food_value += effective_dry * packages[DRY].value;
    }
    
    double other_value = effective_other * packages[OTH].value;
    return food_value + other_value;
}

Solution solve(const ProblemData& problem) {

    auto start_time = chrono::steady_clock::now();
    auto allowed_duration = chrono::milliseconds(long(problem.time_limit_minutes * 60 * 1000));
    auto safe_duration = chrono::duration_cast<chrono::milliseconds>(allowed_duration * 95 / 100);
    auto deadline = start_time + safe_duration;
    

    Solution best_global_solution;
    double best_value = -numeric_limits<double>::max();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    vector<double> starting_ratios = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    for (double start_ratio : starting_ratios) {
        if (chrono::duration_cast<chrono::milliseconds>(deadline - chrono::steady_clock::now()).count() <= 0) break;
        
        
        double dry_ratio = start_ratio;
        double perishable_ratio = 1.0 - start_ratio;
        double learning_rate = 0.2;
        int direction = 1; 

        double previous_value = -numeric_limits<double>::max();
        bool first_iteration = true;
        int no_imp_count = 0;
        const int max_no_improvement = 20; 
        const double min_imp_threshold = 0.01;
        
        double temperature = 100.0;
        double cooling_rate = 0.95;

        while (true) {
            auto current_time = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(deadline - current_time).count() <= 0) {
                break;
            }

            Solution current_solution;
            current_solution.reserve(problem.helicopters.size());

            vector<int> rem_food_demand(problem.villages.size());
            vector<int> rem_other_demand(problem.villages.size());
            for (size_t i = 0; i < problem.villages.size(); ++i) {
                rem_food_demand[i] = 9 * problem.villages[i].population;
                rem_other_demand[i] = problem.villages[i].population;
            }

            for (const auto& helicopter : problem.helicopters) {
                auto now_h = chrono::steady_clock::now();
                if (chrono::duration_cast<chrono::milliseconds>(deadline - now_h).count() <= 0) break;
                
                HelicopterPlan plan;
                plan.helicopter_id = helicopter.id;
                const Point& home = problem.cities[helicopter.home_city_id - 1];

                double current_dist_budget = problem.d_max;
                double avg_food_wt = dry_ratio * problem.packages[DRY].weight + perishable_ratio * problem.packages[PER].weight;

                while (current_dist_budget > 1e-6) {
                    int best_first_vil_idx = -1;
                    double best_init_value = 0;
                    int best_init_dry = 0, best_init_peri = 0, best_init_other = 0;

                    for (size_t i = 0; i < problem.villages.size(); ++i) {
                        auto now_v = chrono::steady_clock::now();
                        if (chrono::duration_cast<chrono::milliseconds>(deadline - now_v).count() <= 0) break;
                        if (rem_food_demand[i] <= 0 && rem_other_demand[i] <= 0) continue;

                        const Point& village_coord = problem.villages[i].coords;
                        double trip_distance = 2.0 * distance(home, village_coord);
                        if (trip_distance > helicopter.distance_capacity || trip_distance > current_dist_budget) continue;

                        if (avg_food_wt < 1e-9) continue;
                        
                        int food_to_send = min(rem_food_demand[i], static_cast<int>(helicopter.weight_capacity / avg_food_wt));
                        int dry_units = static_cast<int>(food_to_send * dry_ratio);
                        int perishable_units = food_to_send - dry_units;

                        double food_weight = dry_units * problem.packages[DRY].weight + perishable_units * problem.packages[PER].weight;
                        if (food_weight > helicopter.weight_capacity + 1e-9) continue;

                        double rem_weight = helicopter.weight_capacity - food_weight;
                        int other_units = 0;
                        if (problem.packages[OTH].weight > 1e-9 && rem_weight > 1e-9) {
                            other_units = min(rem_other_demand[i], static_cast<int>(rem_weight / problem.packages[OTH].weight));
                            other_units = max(0, other_units);
                        }
                        
                        double total_weight = food_weight + other_units * problem.packages[OTH].weight;
                        if (total_weight > helicopter.weight_capacity + 1e-9) continue;

                        double value = calculateVillageValue(problem.villages[i], dry_units, perishable_units, other_units, problem.packages);
                        double cost = helicopter.fixed_cost + helicopter.alpha * trip_distance;
                        double net_value = value - cost;

                        if (net_value > best_init_value) {
                            best_init_value = net_value;
                            best_first_vil_idx = i;
                            best_init_dry = dry_units;
                            best_init_peri = perishable_units;
                            best_init_other = other_units;
                        }
                    }

                    if (best_first_vil_idx == -1) break;

                     Trip current_trip;
                     vector<bool> visited_in_this_trip(problem.villages.size(), false);
                    
                    Drop first_drop = {problem.villages[best_first_vil_idx].id, best_init_dry, best_init_peri, best_init_other};
                    current_trip.drops.push_back(first_drop);
                    visited_in_this_trip[best_first_vil_idx] = true;

                    double current_trip_weight = best_init_dry * problem.packages[DRY].weight + best_init_peri * problem.packages[PER].weight + best_init_other * problem.packages[OTH].weight;
                    Point last_location = problem.villages[best_first_vil_idx].coords;
                    
                    while (true) {
                         int best_next_village_idx = -1;
                         double best_val_added_net = 0;
                         Drop best_next_drop;
                         double best_wt_added = 0;

                        for (size_t j = 0; j < problem.villages.size(); ++j) {
                            auto now_v2 = chrono::steady_clock::now();
                            if (chrono::duration_cast<chrono::milliseconds>(deadline - now_v2).count() <= 0) break;
                            if (visited_in_this_trip[j] || (rem_food_demand[j] <= 0 && rem_other_demand[j] <= 0)) continue;

                            const Point& next_village_coord = problem.villages[j].coords;
                            double distance_added = distance(last_location, next_village_coord) + distance(next_village_coord, home) - distance(last_location, home);
                            
                             double total_trip_dist_if_added = 0;
                             Point temp_loc = home;
                             for(const auto& drop : current_trip.drops) {
                                 total_trip_dist_if_added += distance(temp_loc, problem.villages[drop.village_id-1].coords);
                                 temp_loc = problem.villages[drop.village_id-1].coords;
                             }
                             total_trip_dist_if_added += distance(temp_loc, next_village_coord) + distance(next_village_coord, home);

                            if (total_trip_dist_if_added > helicopter.distance_capacity || total_trip_dist_if_added > current_dist_budget) continue;

                            double remaining_weight_cap = helicopter.weight_capacity - current_trip_weight;
                            if (remaining_weight_cap <= 1e-9) continue;
                            
                            int food_to_send = min(rem_food_demand[j], static_cast<int>(remaining_weight_cap / (avg_food_wt + 1e-9)));
                            int dry_units = static_cast<int>(food_to_send * dry_ratio);
                            int perishable_units = food_to_send - dry_units;

                            double food_weight = dry_units * problem.packages[DRY].weight + perishable_units * problem.packages[PER].weight;
                            if (food_weight > remaining_weight_cap + 1e-9) continue;

                            double temp_rem_weight = remaining_weight_cap - food_weight;
                            int other_units = 0;
                            if (problem.packages[OTH].weight > 1e-9 && temp_rem_weight > 1e-9) {
                                other_units = min(rem_other_demand[j], static_cast<int>(temp_rem_weight / problem.packages[OTH].weight));
                                other_units = max(0, other_units);
                            }
                            
                            double total_weight = food_weight + other_units * problem.packages[OTH].weight;
                            if (total_weight > remaining_weight_cap + 1e-9) continue;
                            
                            if (dry_units + perishable_units + other_units == 0) continue;

                            double value_added = calculateVillageValue(problem.villages[j], dry_units, perishable_units, other_units, problem.packages);
                            double cost_added = helicopter.alpha * distance_added;
                            
                             if (value_added - cost_added > best_val_added_net) {
                                 best_val_added_net = value_added - cost_added;
                                 best_next_village_idx = j;
                                 best_next_drop = {problem.villages[j].id, dry_units, perishable_units, other_units};
                                 best_wt_added = total_weight; 
                             }
                        }

                        if (best_next_village_idx != -1) {
                            current_trip.drops.push_back(best_next_drop);
                            visited_in_this_trip[best_next_village_idx] = true;
                            current_trip_weight += best_wt_added;
                            last_location = problem.villages[best_next_village_idx].coords;
                        } else {
                            break;
                        }
                    }

                     current_trip.dry_food_pickup = 0;
                     current_trip.perishable_food_pickup = 0;
                     current_trip.other_supplies_pickup = 0;
                     
                     for (const auto& drop : current_trip.drops) {
                         current_trip.dry_food_pickup += drop.dry_food;
                         current_trip.perishable_food_pickup += drop.perishable_food;
                         current_trip.other_supplies_pickup += drop.other_supplies;
                         
                         int village_idx = drop.village_id - 1;
                         rem_food_demand[village_idx] = max(0, rem_food_demand[village_idx] - (drop.dry_food + drop.perishable_food));
                         rem_other_demand[village_idx] = max(0, rem_other_demand[village_idx] - drop.other_supplies);
                     }

                    double final_trip_dist = 0;
                    Point current_loc = home;
                    for(const auto& drop : current_trip.drops) {
                        final_trip_dist += distance(current_loc, problem.villages[drop.village_id-1].coords);
                        current_loc = problem.villages[drop.village_id-1].coords;
                    }
                    final_trip_dist += distance(current_loc, home);

                    plan.trips.push_back(current_trip);
                    current_dist_budget -= final_trip_dist;
                }
                current_solution.push_back(plan);
            }
            
            double tval_gained = 0;
            double ttrip_cost = 0;
            vector<double> food_delivered(problem.villages.size() + 1, 0.0);
            vector<double> other_delivered(problem.villages.size() + 1, 0.0);

            for (const auto& helicopter_plan : current_solution) {
                const auto& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
                Point home_city_coords = problem.cities[helicopter.home_city_id - 1];

                for (const auto& trip : helicopter_plan.trips) {
                    if (trip.drops.empty()) continue;

                     double trip_distance = 0;
                     Point current_location = home_city_coords;
                     for (const auto& drop : trip.drops) {
                         const auto& village_coords = problem.villages[drop.village_id - 1].coords;
                         trip_distance += distance(current_location, village_coords);
                         current_location = village_coords;
                     }
                     trip_distance += distance(current_location, home_city_coords);

                    ttrip_cost += helicopter.fixed_cost + (helicopter.alpha * trip_distance);

                    for (const auto& drop : trip.drops) {
                        const auto& village = problem.villages[drop.village_id - 1];
                        double max_food_needed = village.population * 9.0;
                        double food_room_left = max(0.0, max_food_needed - food_delivered[village.id]);
                        double food_in_this_drop = drop.dry_food + drop.perishable_food;
                        double effective_food = min(food_in_this_drop, food_room_left);
                        
                        double effective_vp = min((double)drop.perishable_food, effective_food);
                        tval_gained += effective_vp * problem.packages[PER].value;
                        double effective_vd = min((double)drop.dry_food, effective_food - effective_vp);
                        tval_gained += effective_vd * problem.packages[DRY].value;
                        food_delivered[village.id] += food_in_this_drop;

                        double max_other_needed = village.population * 1.0;
                        double other_room_left = max(0.0, max_other_needed - other_delivered[village.id]);
                        double effective_vo = min((double)drop.other_supplies, other_room_left);
                        tval_gained += effective_vo * problem.packages[OTH].value;
                        other_delivered[village.id] += drop.other_supplies;
                    }
                }
            }
            double final_value = tval_gained - ttrip_cost;

            if (final_value > best_value) {
                double improvement = final_value - best_value;
                best_value = final_value;
                best_global_solution = current_solution;
                
                if (improvement >= min_imp_threshold) {
                    no_imp_count = 0;
                } else {
                    no_imp_count++;
                }
            } else {
                no_imp_count++;
            }
            
            if (no_imp_count >= max_no_improvement) {
                break;
            }

            if (first_iteration) {
                previous_value = final_value;
                first_iteration = false;
                dry_ratio += direction * learning_rate;
            } else {
                double acceptance_prob = exp((final_value - previous_value) / temperature);
                
                if (final_value < previous_value) {
                    if (dis(gen) < acceptance_prob) {
                        previous_value = final_value;
                        dry_ratio += direction * learning_rate;
                    } else {
                        direction *= -1;
                        learning_rate *= 0.9;
                        previous_value = final_value;
                        dry_ratio += direction * learning_rate;
                    }
                } else {
                    learning_rate *= 1.1;
                    previous_value = final_value;
                    dry_ratio += direction * learning_rate;
                }
            }
            
            dry_ratio = max(0.0, min(1.0, dry_ratio));
            perishable_ratio = 1.0 - dry_ratio;
            learning_rate = max(0.05, min(0.3, learning_rate));
            temperature *= cooling_rate; 

        }
    }

    
    Solution validated_solution;
    for (const auto& plan : best_global_solution) {
        HelicopterPlan validated_plan;
        validated_plan.helicopter_id = plan.helicopter_id;
        
        const auto& helicopter = problem.helicopters[plan.helicopter_id - 1];
        const Point& home = problem.cities[helicopter.home_city_id - 1];
        double total_distance_used = 0.0;
        
        for (const auto& trip : plan.trips) {
            if (trip.drops.empty()) continue;
            
            int total_dry_dropped = 0, total_peri_dropped = 0, total_other_dropped = 0;
            for (const auto& drop : trip.drops) {
                total_dry_dropped += drop.dry_food;
                total_peri_dropped += drop.perishable_food;
                total_other_dropped += drop.other_supplies;
            }
            
            if (trip.dry_food_pickup != total_dry_dropped || trip.perishable_food_pickup != total_peri_dropped || trip.other_supplies_pickup != total_other_dropped || trip.dry_food_pickup < 0 || trip.perishable_food_pickup < 0 || trip.other_supplies_pickup < 0) {
                continue; 
            }
            
            bool has_negative_drops = false;
            for (const auto& drop : trip.drops) {
                if (drop.dry_food < 0 || drop.perishable_food < 0 || drop.other_supplies < 0) {
                    has_negative_drops = true;
                    break;
                }
            }
            if (has_negative_drops) continue;
            
            double trip_distance = 0.0;
            Point current_location = home;
            for (const auto& drop : trip.drops) {
                const Point& village_coords = problem.villages[drop.village_id - 1].coords;
                trip_distance += distance(current_location, village_coords);
                current_location = village_coords;
            }
            trip_distance += distance(current_location, home);
            
            if (trip_distance > helicopter.distance_capacity + 1e-9) continue;
            
            if (total_distance_used + trip_distance > problem.d_max + 1e-9) continue;
            
            double trip_weight = trip.dry_food_pickup * problem.packages[DRY].weight + trip.perishable_food_pickup * problem.packages[PER].weight + trip.other_supplies_pickup * problem.packages[OTH].weight;
            if (trip_weight > helicopter.weight_capacity + 1e-9) continue;
            
            validated_plan.trips.push_back(trip);
            total_distance_used += trip_distance;
        }
        
        if (!validated_plan.trips.empty()) {
            validated_solution.push_back(validated_plan);
        }
    }

    return validated_solution;
}
