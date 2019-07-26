Developer Notes
===============

In this part, we provide detailed explanation on how certain elements of the computational model are implemented in the current version of the code. There can be multiple ways of casting a given computation model in code. Here we specify programming choices made when implementing certain aspects of the model and provide logic and specification.


Version 0.2
***********

Types
-----

A new extension of the model includes the introduction of unobserved heterogeneity in the form of discrete mass points
in the disutility of work. We refer to the existence of a fixed number of individual types in the model. We estimate
the frequencies of each type in the population as well as the disutility of part-time and full-time work, `theta_p`
and `theta_f`, for all types compared to the baseline type (coefficient equal zero).

Note the convention for the specification of the `init_file` after the introduction of types:

* One type: `theta_pi`, `theta_fi`, `share_i` are not included in the initialisation file:
    * baseline type with coefficients `theta_p` and `theta_f` equal to zero, which occurs with zero probability 1.
    * above implies that the coefficients for the baseline type equal the constant.
* More types: `theta_pi`, `theta_fi`, `share_i`, for each `i` from 1 to the number of types including.


Version 0.1
************

pyth_create_state_space
-----------------------

Functionality
"""""""""""""
The function creates state space related objects given state space components.

Inputs
""""""
The inputs of the function come from the model specification. They consist of necessary information on the initial conditions and on the dimension of the state space components. In the soepy toy model, the initial condition are the completed years of education. The function requires the range of years of education completed from the (simulated) sample as input. It further requires the number of periods in the model and the number of discrete choices individuals face each period.

Outputs
"""""""
The tate space related outputs of the function are collected in the array "args". "args" contains:

* states_all: an array of tuples of state space components that constitute an admissible state space point each and taken together represent the entirity of admissible states across all periods in the model
* states_number_period: a one dimensional array (i.e., a vector) that collects the number of admissible state space points for each period
* mapping_state_index: an array which maps each state recorded in states all to a unique integer identifier
* max_states_period: a scalar value corresponding to the maximum of the number of states per period over all periods

Code content
""""""""""""

In the following the way the function is programmed to work is described in more detail.

First, we create array containers for the states_all and mapping_state_index arrays. In the loop executed further below as a part of this function, we want to populate these arrays with the state space components. The components of our toy model state space are:

* the time variable: period
* the initial conditions: education
* the history of choices: choice_lagged
* the accumulated experience in part-time employment: exp_p
* the accumulated experience in full-time employment: exp_f

Initially, the containers are filled with missing values. The important part in initializing the containers is to specify the correct dimensions.

Let us briefly discuss the dimension of the mapping_state_index array. Each dimension component corresponds to one state space component we would like to keep record of. In the funtion loop, a unique tuple of state space component values is assigned a unique integer identifier k. The dimensions correspond to:

* dimension 1 num_periods to the period number
* dimension 2 educ_range to the years of education
* dimension 3 num_choices to the choice of the individual
* dimension 4 num_periods to years of experience in part-time
* dimension 5 num_periods to years of experience in full-time

Let us briefly discuss the dimention of the states_all array:

* the 1st dimension is determined by the number of periods in our model
* the 2nd dimension is related to the maximum number of state space points ever feasible / possibly reachable in one of the periods. 
* the 3rd dimension is equal to the number of remaining state space components (except period number) that we want to record: educ_years + educ_min, choice_lagged, exp_p, exp_f

We note that we set the size of dimension to equal to the highest possible number of state space points ever reachable in a period. This number is calculated as the maximum number of combinantions of the state space componenents given no restrictions.

Now we can loop through all admissible state space points and fill up the constructed arrays with necessary information.

In the following, we note how education is introduced as an initial condition in the loop:

If we want to record only admissible state space points, we have to account for the fact that individuals in the model start making labor supply choices only after they have completed education. As a reminder, we note that we model individuals from age 16 (legally binding mininum level of education) until age 60 (typical retirement entry age for women), i.e., for 45 periods, where the 1st period corresponds to age 16, the second period to age 17, etc. The current specification of starting values is equivalent to the observation / simulated reality that individuals in the sample have completed something between 10 and 14 years of edcuation. In our loop we want to take into account the fact that, in the first period at age 16, only individuals who have completed 10 years of education (assuming education for everyone starts at the age of 6) will be making a labor market choice between full-time (F), part-time (P), and non-employment (N). The remaining individuals are still in education, such that a state space point where years of education equal e.g. 11 and a labor market choice of e.g. part-time is observed is not admissible and should therefore not be recorded. This is ensured by the if clause "educ_years > period".

When starting the model in the way described above, we need to account for an additional state space point which corresponds to individuals who have just entered the model. Let us consider the very first period, period 0 in the loop, as an example. In this period, only individuals with 10 years of education enter the model and make their first labor supply choice. In order to later be able to compute the instataneous utility for this individuals, we need to record their state space components. As there is no lagged choice available to record, we force set the lagged choice component to zero. In particular, for states for which the condition (period == educ_years) is true, i.e., for individuals who have just entered the model, we record a state: (educ_years + educ_min, 0, 0, 0).

Finally, we briefly repeat what has been recorded in one of the main function output objects, the states_all array:

* educ_years + educ_min: in this example, values from 10 to 14
* choice_lagged: 0, 1, 2 corresponding to N, P, and F
* exp_p: part-time experience that can range from 0 to 9
* exp_f: full-time experience that can range from 0 to 9
 
Note: There is a difference to respy here. In respy, the loop in experience is one iteration longer, goes to num_periods + 1 instead of to num_periods.





pyth_backward_induction
-----------------------

Functionality
"""""""""""""
Solves the dynamic discrete choice model in a backward induction procedure, in which the error terms are integrated out in a Monte Carlo simulation procedure. Obtaining a solution means making such a choice at each state space point as to obtain the highest of the choice specific value funftions. The current period value function is then the sum of current periods flow utility of the optimal choice and next period's flow utility given an optimal choice in the future.

Inputs
""""""
In a final version of soepy, the pyth_create_state_space function is called before the backward induction procedure. The backward induction procedure needs the outputs of the pyth_create_state_space function as inputs. It further relies on multiple inputs from the model specification: num_periods, num_choices, educ_max, educ_min, educ_range, mu, delta,o ptim_paras, num_draws_emax, seed_emax, shocks_cov.

Outputs
"""""""
The array periods_emax contains the highest value function value among the choice specific value function values for the three labor market choices at each admissible state space point in each period of the model. The array is of dimension number periods by maximum number of admissible choices over all periods.


Code content
""""""""""""
The individuals in our model solve their optimization problem by making a labor supply choice in every period. They choose the option that is associated with the highest value function. The value function for each of the 3 alternatives is the sum of the current period flow utility of choosing alternative j and a continuation value. The continuation value is, in turn, next period's value function given an optimal choice in the future. 

The flow utility includes the current period wage shock, which the individual becomes aware of in the begining of the period and includes in her calculations. To obtain an estimate of the continuation value the individual has to integrate out the distribution of the future shocks. In the model implementation, we perform numerical integration via a Monte Carlo simulation.

In the function, we generate draws from the error term distribution defined by the error term distribution paramters in the model specification, (in the current model spesification - shocks_cov). For each peiod we draw as many disturbances as num_draws_emax. These draws let us numerically integrate out the error term in a Monte Carlo simulation procedure. This is necessary for computing the continuation values and, ultimately, the value functions and the model's solution. Integating out the error term, represents the process in which individuals in the model form expectations about the future. Assuming rational expectations and a known error term distribution up to its parameters, individuals take the possible realization of the error terms into account by computing the expected continuation values over the distribution of the errors. For every period, we simmulate num_draws_emax draws from the error term distribution.

In the current formulation, we assume that the wage process is subject to additive measurement error. The disturbances for non-employment, the part-time, and the full-time wage are normally distributed with mean zero. The spesification assumes no serial and also no contemporaneous correlation across the error terms.

Before we begin the backward iteration procedure, we initialize the container for the final result. It is the array periods_emax with dimensions number of periods by maximum number of admissible states (max_states_period).

The backward induction loop calls several functions defined separately. As the name suggest, we loop backwards:

* construct_covariates: determines the education level given the state space component years of education
* construct_emax: integrates out the error term by averaging the value function values over the drawn realization of the error term. In this, the value function is computed using furter nested functions.
* calculate_utilities: calculates the flow utility using the systematic wage (wage without error), the period wage (systematic wage and error), the consumption utility (first part of the utility function), and total utility (consumption utility and U(.)).
* calculate_continuation_values: recursively obtains a continuation value given period and state. The function selects the relevant element of the periods_emax array given period number and state space components. This is possible since the whole loop is executed backwards.

Note concerning calculate_consumption_utilities:

In the toy model, consumption in any period is zero if the individual chooses non-employment. This is the case because consumption is simply the product of the period wage and the hours worked, and the hours worked in the case of non-employment are equal to zero. The calculation of the 1st part of the utility function related to consumption involves taking period consumption to the negative pover mu. In the programm, this would yield -inf. To avoid this complication, here the consumption utility of non-employment is normalized to zero.


pyth_simulate
-------------

Functionality
"""""""""""""
Simulate a data set given model spesification.

Inputs
""""""
In a final version of soepy, the functions pyth_create_state_space and pyth_backward_induction are called before the backward_induction procedure. The simulation procedure requires the outputs of the former functions as inputs. It further relies on multiple inputs from the model specification. Most are the same as the ones required by the backward induction procedure: num_periods, num_choices, educ_max, educ_min, educ_range, mu, delta,o ptim_paras, num_draws_emax, seed_emax, shocks_cov. In addition, num_agents_sim and seed_sim are also required.

Outputs
"""""""
A pandas data frame with infromation about agents experiences in each period such as the choice, wage, flow utility, etc.

First, we need to genrate draws of the error term distribution. We note that this set of draws is different to the one used in the backward induction procedure. In the simulation, we need another set of draws to represent our simulated reality. In our model, at the beginning of every new period, individuals are hit by a productivity shock. They are aware of the realization of the shock when making their labor supply choice for the period. For every period, we simmulate num_agents_sim draws of the error term distribution.

Next, we need to simulate a sample of initial conditions. In this example, we need to assing a value for the years of education to every agent whose life-cycle we want to simulate.

Finally, we loop forward through all agents and all periods to generate agent's experiences in the model and record these in a data frame. 

We note that the sumulation procedure uses a slightly modified verion of the construct covariates function than the backward iteration procedure. During backward iteration, the education level is determined for all possible years of education depending on which state space point has currently been reached by the loop. During simulation, the education level needs to be determined according the simulated initial condition for the individual currently reached by the loop. We further note that the simulation procedure does need all subfunctions related to the calculation of the instantaneous utility, but it does not need the construction of the expected maximum (construct_emax) as a subfunction. The model's solution has already been computed in the backward iteration procedure. During simulation, we can access the relevant continuation value recorded in the periods_emax array given the current period number and state space components determined by the agent's experiences so far.  

Possibles To Do's to consider:
* rearrange init file as to possibly avoid selecting parameters using big numbers
* consider wether it is more efficient to generate education level information at some other stage of the core (currently in loop over state space components as part of the construct covariates function)
* add mean as function input to error term generation
* move hours definition from calculate_consumption_utilties in shared_auxiliary to the initialization file.

