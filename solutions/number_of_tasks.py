# number of tasks for each local training step: training + prediction + evaluation
number_of_local_tasks_for_one_round_and_one_orga = 3
# number of tasks for each training step on all orgs: local_tasks * n_orgs + aggregation_task
n_orgs = 2
number_of_tasks_per_round = number_of_local_tasks_for_one_round_and_one_orga * n_orgs + 1
# number of tasks for regular rounds: tasks_per_round * number_rounds
number_of_tasks_for_intermediate_rounds = number_of_tasks_per_round * 3
# number of initialization tasks: one on each data organization
number_of_initialization_tasks = 2
# number of finalization tasks: last local retraining on each data org
number_of_finalization_tasks = n_orgs * 3
# total number of tasks in the compute plan
number_of_initialization_tasks + number_of_tasks_for_intermediate_rounds + number_of_finalization_tasks